import cv2
import numpy as np

from cnn_utils import classification_utils, image_utils, aug_utils


# import mnist_data_utils


def erode_batch(X, ks):
	ks = int(ks)
	n = X.shape[0]
	h = X.shape[1]
	w = X.shape[2]
	c = X.shape[3]
	X_temp = np.transpose(X, (1, 2, 3, 0))
	X_temp = np.reshape(X_temp, (h, w, -1))
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
	#	X_temp = cv2.resize( X_temp, None, fx=scale_factor, fy=scale_factor )
	X_temp = cv2.erode(X_temp, kernel)
	h_new = X_temp.shape[0]
	w_new = X_temp.shape[1]
	X_out = np.reshape(X_temp, (h_new, w_new, c, n))
	X_out = np.transpose(X_out, (3, 0, 1, 2))
	return X_out

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


def gen_batch(ims_data, labels_data,
              batch_size, target_size=None, randomize=False,
              normalize_tanh=False,
              labels_to_onehot_mapping=None,
              aug_params=None, convert_onehot=False,
              yield_aug_params=False, yield_idxs=False,
              random_seed=None):
	'''

	:param ims_data: list of images, or an image. If a single image, it will be automatically converted to a list
	:param labels_data:
	:param batch_size:
	:param target_size:
	:param randomize:
	:param normalize_tanh:
	:param labels_to_onehot_mapping:
	:param aug_params:
	:param convert_onehot:
	:param yield_aug_params:
	:param yield_idxs:
	:param random_seed:
	:return:
	'''
	if random_seed:
		np.random.seed(random_seed)

	if not isinstance(ims_data, list):
		ims_data = [ims_data]

	if not isinstance(normalize_tanh, list):
		normalize_tanh = [normalize_tanh]
	else:
		assert len(normalize_tanh) == len(ims_data)

	if aug_params is not None:
		if not isinstance(aug_params, list):
			aug_params = [aug_params]
		else:
			assert len(aug_params) == len(ims_data)

	if target_size is not None:
		if not isinstance(target_size, list):
			target_size = [target_size]
		else:
			assert len(target_size) == len(ims_data)


	# if we have labels that we want to generate from,
	# put everything into a list for consistency
	# (useful if we have labels and aux data)
	if labels_data is not None:
		if not isinstance(labels_data, list):
			labels_data = [labels_data]

		# each entry should correspond to an entry in labels_data
		if not isinstance(convert_onehot, list):
			convert_onehot = [convert_onehot]
		else:
			assert len(convert_onehot) == len(labels_data)


	idxs = [-1]

	n_ims = ims_data[0].shape[0]
	h = ims_data[0].shape[1]
	w = ims_data[0].shape[2]

	if target_size is not None:
		# pad each image and then re-concatenate
		ims_data = [np.concatenate([
			image_utils.pad_or_crop_to_shape(x, target_size)[np.newaxis]
			for x in im_data], axis=0) for im_data in ims_data]

	while True:
		if randomize:
			idxs = np.random.choice(n_ims, batch_size, replace=True)
		else:
			idxs = np.linspace(idxs[-1] + 1, idxs[-1] + 1 + batch_size - 1, batch_size, dtype=int)
			restart_idxs = False
			while np.any(idxs >= n_ims):
				idxs[np.where(idxs >= n_ims)] = idxs[np.where(idxs >= n_ims)] - n_ims
				restart_idxs = True

		ims_batches = []
		for im_data in ims_data:
			X_batch = im_data[idxs]

			if not X_batch.dtype == np.float32:
				X_batch = (X_batch / 255.).astype(np.float32)
			if normalize_tanh:
				X_batch = image_utils.normalize(X_batch)

			if aug_params is not None:
				X_batch, aug_params = aug_utils.aug_mtg_batch(X_batch, **aug_params)
			ims_batches.append(X_batch)

		if labels_data is not None:
			labels_batches = []
			for Y in labels_data:
				if convert_onehot:
					Y_batch = classification_utils.labels_to_onehot(
						Y[idxs],
				        label_mapping=labels_to_onehot_mapping)
				else:
					Y_batch = Y[idxs]
				labels_batches.append(Y_batch)
		else:
			labels_batches = None

		if not randomize and restart_idxs:
			idxs[-1] = -1

		if yield_aug_params and yield_idxs:
			yield tuple(ims_batches) +  tuple(labels_batches) + (aug_params, idxs)
		elif yield_aug_params:
			yield tuple(ims_batches) + tuple(labels_batches) + (aug_params, )
		elif yield_idxs:
			yield tuple(ims_batches) + tuple(labels_batches) + (idxs, )
		else:
			yield tuple(ims_batches) + tuple(labels_batches)


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
