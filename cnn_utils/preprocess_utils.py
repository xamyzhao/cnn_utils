import numpy as np
import json
import cv2
import scipy.io as sio

def putGaussianMaps(H,joints,in_range,sigma,crop_size_x,crop_size_y,stride):
	start = stride/2.0 - 0.5
	x = start + np.array(range(H.shape[1])) * stride
	y = start + np.array(range(H.shape[0])) * stride
	xv,yv = np.meshgrid(x,y,sparse=False,indexing='xy')
	
	for i in xrange(n_joints):	
		if(not in_range[i]):
			continue
		d = (xv - joints[i,0]) * (xv - joints[i,0]) + (yv - joints[i,1]) * (yv - joints[i,1])
		exponent = d/(2*sigma*sigma)
		H[:,:,i] = np.exp(-exponent)

	#H[:,:,n_joints] = np.maximum(0, 1 - np.amax(H,axis=2))


def makeCenterMap(H,p,sigma,param):
	if 'label_size' in param:
		kern_size = param['label_size']
	else:
		kern_size = int(round(6*sigma))+1
	kern = np.zeros((kern_size,kern_size))
	kern_center = (kern_size-1)/2

	for i in xrange(kern_size):
		for j in xrange(kern_size):
			kern[i][j] = np.exp(-1.0 * ((i - kern_center)**2 + (j - kern_center)**2)/(2.0*sigma*sigma)) 	

	x0 = int(p[0]) - kern_center
	y0 = int(p[1]) - kern_center

	h = H.shape[0]
	w = H.shape[1]
	for i in xrange(kern_size):
		for j in xrange(kern_size):
			if(x0 + j >= 0 and y0 + i >= 0 and x0 + j < w and y0 + i < h): 
				H[y0+i,x0+j,0] += kern[i][j]

