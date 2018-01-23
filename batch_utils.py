import numpy as np
import sys
import data_utils
import os
import cv2
import scipy
import scipy.io as sio

import vis_utils
import aug_utils
#import mnist_data_utils
import math

def pyrDown_batch( X ):
	return np.transpose( np.reshape( cv2.pyrDown( np.reshape( np.transpose( X, (1,2,3,0) ), X.shape[1:3] + (X.shape[3]*X.shape[0],)) ), (X.shape[1]/2, X.shape[2]/2, X.shape[3], X.shape[0],) ), (3,0,1,2) )

def resize_batch( X, scale_factor): 
	n = X.shape[0]	
	h = X.shape[1]
	w = X.shape[2]
	c = X.shape[3]
	X_temp = np.transpose( X, (1,2,3,0) )
	X_temp = np.reshape( X_temp, (h,w,c*n) )
	X_temp = cv2.resize( X_temp, None, fx=scale_factor, fy=scale_factor )
	h_new = X_temp.shape[0]
	w_new = X_temp.shape[1]
	X_out = np.reshape( X_temp, ( h_new, w_new,c,n) )
	X_out = np.transpose( X_out, (3,0,1,2) )
	return X_out


def gen_generator_batch( batch_gen, include_noise_input = False, noise_size = 100 ):
	# assumes batch_gen returns X,Y
	while(1):
		X,_ = next(batch_gen)
		
		if include_noise_input:
			batch_size = X.shape[0]
			x_noise = np.random.rand( batch_size, noise_size ) * 2.0 - 1.0
			yield [X, x_noise]
		else:
			yield X

#def gen_aug_batch( batch_gen, include_noise_input = False, noise_size = 100 ):
	
def gen_batch( X, Y, batch_size, target_size=None, randomize=False, labels_to_onehot_mapping = None, aug_X = False, aug_X_params = None, convert_onehot = True ):
#	np.random.seed(17)
	idxs = [-1]

	n_ims = X.shape[0]
	h = X.shape[1]
	w = X.shape[2]
	if target_size is None:
		target_size = h

	pad_h = (target_size	- h)/2.
	if pad_h > 0:
		X = np.pad( X, ( (0,0), (int(np.floor(pad_h)), int(np.ceil(pad_h))),(int(np.floor(pad_h)), int(np.ceil(pad_h))), (0,0)), mode='constant')
 
	while True:
		if randomize:
			idxs = (np.random.rand( batch_size ) * (n_ims-1)).astype(int)
 		else:
			idxs = np.linspace(idxs[-1]+1,idxs[-1]+1+batch_size-1, batch_size, dtype=int ) 
			restart_idxs = False
			if np.any(idxs >= n_ims):
				idxs[ np.where( idxs >= n_ims ) ] = idxs[ np.where( idxs >= n_ims ) ]-n_ims
				restart_idxs = True
		X_batch = X[idxs]

		if aug_X:
			X_batch,_ = aug_utils.aug_mtg_batch( X_batch, **aug_X_params )

		if Y is not None:
			if convert_onehot:
				Y_batch = data_utils.labels_to_onehot( Y[idxs], label_mapping = labels_to_onehot_mapping) 
			else:
				Y_batch = Y[idxs]
		else:
			Y_batch = None

		if not randomize and restart_idxs:
			idxs[-1] = -1
		yield X_batch, Y_batch

def make_aug_batch( X ):
	X = data_utils.inverse_normalize(X)
	for i in range(X.shape[0]):
		X[i] = aug_im( X[i] )
	return data_utils.normalize( X )

def make_siamese_batch( source_gen, target_gen, lpat_model, batch_size, source_shape, target_shape, write_examples = False, randomize = False ):
	X_A, _ = next( source_gen )
	X_B, _ = next( source_gen )

	for i in range(batch_size):
		while np.array_equal( X_A[i], X_B[i] ):
			X_temp,_ = next(source_gen)
			X_B[i] = X_temp[0]
	return X_A, X_B


if __name__ == '__main__':
#	gen = gen_disc_example( (28,28) )
#	for i in range(50):
#		next(gen)
	#print( gen_flow_batch((128,128,3)) )
	print( pyrDown_batch( np.zeros((8,128,128,3))).shape)
