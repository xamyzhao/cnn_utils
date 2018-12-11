import cv2
import numpy as np

from cnn_utils import image_utils


# import mnist_data_utils


def dilate_batch(X, ks):
	ks = int(ks)
	n = X.shape[0]
	h = X.shape[1]
	w = X.shape[2]
	c = X.shape[3]
	X_temp = np.transpose(X, (1, 2, 3, 0))
	X_temp = np.reshape(X_temp, (h, w, -1))
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
	#	X_temp = cv2.resize( X_temp, None, fx=scale_factor, fy=scale_factor )
	X_temp = cv2.dilate(X_temp, kernel)
	h_new = X_temp.shape[0]
	w_new = X_temp.shape[1]
	X_out = np.reshape(X_temp, (h_new, w_new, c, n))
	X_out = np.transpose(X_out, (3, 0, 1, 2))
	return X_out


def gaussianBlur_batch(X, sigma):
	n = X.shape[0]
	h = X.shape[1]
	w = X.shape[2]
	c = X.shape[3]
	X_temp = np.transpose(X, (1, 2, 3, 0))
	X_temp = np.reshape(X_temp, (h, w, c * n))
	#	X_temp = cv2.resize( X_temp, None, fx=scale_factor, fy=scale_factor )
	X_temp = cv2.GaussianBlur(X_temp, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
	h_new = X_temp.shape[0]
	w_new = X_temp.shape[1]
	X_out = np.reshape(X_temp, (h_new, w_new, c, n))
	X_out = np.transpose(X_out, (3, 0, 1, 2))
	return X_out


def pyrDown_batch(X):
	return np.transpose(
		np.reshape(cv2.pyrDown(np.reshape(np.transpose(X, (1, 2, 3, 0)), X.shape[1:3] + (X.shape[3] * X.shape[0],))),
		           (X.shape[1] / 2, X.shape[2] / 2, X.shape[3], X.shape[0],)), (3, 0, 1, 2))


def resize_batch(X, scale_factor):
	n = X.shape[0]
	h = X.shape[1]
	w = X.shape[2]
	c = X.shape[3]
	X_temp = np.transpose(X, (1, 2, 3, 0))
	X_temp = np.reshape(X_temp, (h, w, c * n))
	if np.max(X_temp) <= 1.0:
		X_temp = cv2.resize(X_temp * 255, None, fx=scale_factor, fy=scale_factor) / 255.
	else:
		X_temp = cv2.resize(X_temp, None, fx=scale_factor, fy=scale_factor)
	h_new = X_temp.shape[0]
	w_new = X_temp.shape[1]
	X_out = np.reshape(X_temp, (h_new, w_new, c, n))
	X_out = np.transpose(X_out, (3, 0, 1, 2))
	return X_out


def pad_or_crop_to_shape(X, out_shape, border_color=1.):
	border_size = (out_shape[0] - X.shape[1], out_shape[1] - X.shape[2])
	X = np.pad(X,
	           ((0, 0),
	            (int(np.floor(border_size[0] / 2.)), int(np.ceil(border_size[0] / 2.))),
	            (int(np.floor(border_size[1] / 2.)), int(np.ceil(border_size[1] / 2.))),
	            (0, 0)),
	           mode='constant',
	           constant_values=border_color)
	return X


def gen_generator_batch(batch_gen, include_noise_input=False, noise_size=100):
	# assumes batch_gen returns X,Y
	while (1):
		X, _ = next(batch_gen)

		if include_noise_input:
			batch_size = X.shape[0]
			x_noise = np.random.rand(batch_size, noise_size) * 2.0 - 1.0
			yield [X, x_noise]
		else:
			yield X


# def gen_aug_batch( batch_gen, include_noise_input = False, noise_size = 100 ):


def make_aug_batch(X):
	X = image_utils.inverse_normalize(X)
	for i in range(X.shape[0]):
		X[i] = aug_im(X[i])
	return image_utils.normalize(X)



if __name__ == '__main__':
	#	gen = gen_disc_example( (28,28) )
	#	for i in range(50):
	#		next(gen)
	# print( gen_flow_batch((128,128,3)) )
	print(pyrDown_batch(np.zeros((8, 128, 128, 3))).shape)
