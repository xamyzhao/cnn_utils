import numpy as np
import cv2
import math
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator, interp2d, RectBivariateSpline
from scipy.ndimage import map_coordinates

def augScale(I, points=None, scale_rand=None, obj_scale = 1.0, target_scale = 1.0, pad_value=None, border_color=(0,0,0) ):
	if scale_rand is not None:
	  scale_multiplier = scale_rand
	else:
	  scale_multiplier = randScale(0.8,1.2)

	scale = target_scale/obj_scale * scale_multiplier
	I_new = cv2.resize(I,(0,0),fx=scale,fy=scale)

	target_size = I.shape[0]
	border_size = (target_size - I_new.shape[0], target_size - I_new.shape[1] )
	if border_size[0] > 0:
		I_new = cv2.copyMakeBorder(I_new, int( math.floor( border_size[0]/2.0 )), \
																 	int(math.ceil( border_size[0]/2.0 )),0,0,\
																	cv2.BORDER_CONSTANT, value=border_color)
	elif border_size[0] < 0:
		I_new = I_new[ -int(math.floor( border_size[0]/2.0 )): I_new.shape[0] + int(math.ceil( border_size[0]/2.0 )),:,: ]

	if border_size[1] > 0:
		I_new = cv2.copyMakeBorder( I_new,0,0,int(math.floor( border_size[1]/2.0 )), \
																	int(math.ceil( border_size[1]/2.0 )), \
																 	cv2.BORDER_CONSTANT, value=border_color )
	elif border_size[1] < 0:
		I_new = I_new[:, -int(math.floor( border_size[1]/2.0 )): I_new.shape[1] + int(math.ceil( border_size[1]/2.0 )),: ]

	if points is not None:
	  points = points * scale
	return I_new, points

def augRotate(I, points=None, max_rot_degree=30.0, crop_size_x=None, crop_size_y=None, degree_rand=None, border_color=(0,0,0)):
	if crop_size_x is None:
	  crop_size_x = I.shape[1]
	if crop_size_y is None:
	  crop_size_y = I.shape[0]

	if degree_rand is not None:
	  degree = degree_rand
	else:
	  degree = randRot( max_rot_degree )

	h = I.shape[0]
	w = I.shape[1]

	center = ( (w-1.0)/2.0, (h-1.0)/2.0 )
	R = cv2.getRotationMatrix2D(center,degree,1)
	I = cv2.warpAffine(I,R,(crop_size_x,crop_size_y),borderValue=border_color)

	if points is not None:
		for i in xrange(points.shape[0]):
		  points[i,:] = rotatePoint(points[i,:],R)

	return I, points, degree

def rotatePoint(p,R):
	x_new = R[0,0] * p[0] + R[0,1] * p[1] + R[0,2]
	y_new = R[1,0] * p[0] + R[1,1] * p[1] + R[1,2]
	return np.array((x_new,y_new))

def augCrop(I,joints,obj_pos, param, shift_px=2.0, rand_shift = None):
	crop_size_x = param['crop_size_x']
	crop_size_y = param['crop_size_y']

	if rand_shift is not None:
	  x_shift = rand_shift[0]
	  y_shift = rand_shift[1]
	else:
	  x_shift, y_shift = randShift( shift_px )
	  x_offset = (crop_size_x-1.0)/2.0 - (obj_pos[0] + x_shift)
	y_offset = (crop_size_y-1.0)/2.0 - (obj_pos[1] + y_shift)

	T = np.float32([[1,0,x_offset],[0,1,y_offset]])
	I = cv2.warpAffine(I,T,(crop_size_x,crop_size_y))

	joints[:,0] += x_offset
	joints[:,1] += y_offset
	obj_pos[0] += x_offset
	obj_pos[1] += y_offset

	return I,joints,obj_pos

aug_spaces = [(cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR, None), (cv2.COLOR_BGR2YCR_CB, cv2.COLOR_YCR_CB2BGR, None),(None,None,None)]

def augSaturation( I, aug_percent = 0.2 ):
	rand_space = aug_spaces[ int(round(np.random.rand(1)[0]*len(aug_spaces)-1) )]

	aug_channels = rand_space[2]
	if aug_channels is None:
		aug_channels = rand_channels()# int(round(np.random.rand(1)[0]*2))
	
	for chan in aug_channels:
		I = scaleImgChannel( I, rand_space[0], rand_space[1], chan, aug_percent )
	return I

def augBlur(I, max_blur_sigma = 10.0 ):
	blur_sigma = int(np.random.rand(1)[0]*max_blur_sigma)
	if blur_sigma > 0:
		kernel_size = int(blur_sigma/5.0)*2+1
		I = cv2.GaussianBlur( I, (kernel_size, kernel_size), blur_sigma )
	return I

def rand_channels():
	# select 1-3 channels randomly from the 3 channels available
	return np.random.choice( range(3), int(1+np.random.rand(1)*2), replace=False)

def augNoise(I, max_noise_sigma = 0.1 ):
	noise_sigma = abs((np.random.randn(1)[0] - 0.5)*2 * max_noise_sigma)
	
	rand_space = aug_spaces[ int(round(np.random.rand(1)[0]*len(aug_spaces)-1) )]
	chans = rand_channels()

	I = I.astype(np.float32)

	if rand_space[0] is not None:
		I = cv2.cvtColor( I, rand_space[0] )

	noise = np.zeros( I.shape, dtype=np.float32 )

	for chan in chans:
		noise_sigma = min( 0.05*(np.max(I[:,:,chan])-np.min(I[:,:,chan])), noise_sigma )
		noise[:,:,chan] = np.multiply(np.random.randn( I.shape[0], I.shape[1] ), noise_sigma)

	if rand_space[1] is not None:
		I = cv2.cvtColor( I, rand_space[1] )

	I = np.clip( np.add(I, noise),0,1.0)
	return I

def scaleImgChannel(I, fwd_color_space = cv2.COLOR_BGR2HSV, bck_color_space=cv2.COLOR_HSV2BGR, channel=None, aug_percent = 0.2 ):
	if np.max(I) > 1.0 and not I.dtype == np.float32:
		I = np.multiply(I.astype(np.float32), 1/255.0)
	I = I.astype(np.float32)
	if fwd_color_space is not None:
		I = cv2.cvtColor( I, fwd_color_space )
	s_scale = randScale( 1.0 - aug_percent, 1.0 + aug_percent )
	I[:,:,channel] = np.multiply( I[:,:,channel], s_scale )
	if fwd_color_space is not None:
		I = cv2.cvtColor( I, bck_color_space )
	
	return I

def swapLeftRight(joints):
	right = [3,4,5,9,10,11]
	left = [6,7,8,12,13,14]

	for i in xrange(6):
	  ri = right[i]-1
	  li = left[i]-1
	  temp_x = joints[ri,0]
	  temp_y = joints[ri,1]
	  joints[ri,:] = joints[li,:]
	  joints[li,:] = np.array([temp_x,temp_y])

	return joints

def augFlip(I,joints=None, obj_pos=None,flip_rand=None ):
	if flip_rand is not None:
	  do_flip = flip_rand
	else:
	  do_flip = randFlip()
	if(do_flip):
		I = np.fliplr(I)
		if obj_pos is not None:
			obj_pos[0] = I.shape[1] - 1 - obj_pos[0]
		if joints is not None:
			joints[:,0] = I.shape[1] - 1 - joints[:,0]
			joints = swapLeftRight(joints)
	return I,joints,obj_pos

def randScale( scale_max, scale_min ):
	rnd = np.random.rand()
	return ( scale_max - scale_min ) * rnd + scale_min

def randRot( max_rot_degrees ):
	return (np.random.rand()-0.5)*2 * max_rot_degrees

def randShift( shift_px ):
	x_shift = int(shift_px * (np.random.rand()-0.5))
	y_shift = int(shift_px * (np.random.rand()-0.5))
	return x_shift, y_shift

def randFlip():
	return np.random.rand() < 0.5

def inRange( joints, I, in_range):
	minLoc = 2
	for i in xrange(n_joints):
	  if(joints[i,0] < minLoc or joints[i,1] < minLoc or
		  joints[i,0] >= I.shape[1] or joints[i,1] >= I.shape[0]):
		 in_range[i] = False
	return in_range

def augProjective(I, max_theta = [15.,15.,15.], scale=1., max_shear = 0.2 ):
	img_shape = I.shape
	if not type(max_theta) == list:
		max_theta = np.asarray( [max_theta]*3 )
	I_in = I.astype(np.float32)

	I_out = np.zeros( img_shape, dtype=np.float32 )
	theta = np.random.rand( 3 )*max_theta*2.0 - max_theta
	theta = theta*math.pi/180.0

	h = img_shape[0]
	w = img_shape[1]
	s = 1.
	R_x = np.asarray( [ [1., 0. ,0.],
										 	[0., s*np.cos(theta[0]), s*np.sin(theta[0])], 
											[0., -s*np.sin(theta[0]), s*np.cos(theta[0])] ], dtype=np.float32 )
	R_y = np.asarray( [	[s*np.cos(theta[1]),0, -s*np.sin(theta[1])], 
											[0., 1., 0.], 
											[ s*np.sin(theta[1]), 0., s*np.cos(theta[1])] ], dtype=np.float32)
	R_z = np.asarray( [	[np.cos(theta[2]), np.sin(theta[2]), 0.],
					 						[-np.sin(theta[2]), np.cos(theta[2]), 0.], 
											[0., 0., 1.] ], dtype=np.float32)
	R  = np.matmul( R_y, R_x )
	R = np.matmul( R_z, R )

 
	t = np.zeros((3,1))
	t[2] = img_shape[0]
	RT = np.concatenate( [R,t], axis=1 )
	
	xv, yv, zv = np.meshgrid( np.linspace( -w/2,w/2, w), np.linspace( -h/2,h/2,h), 0. )

	xyz = np.concatenate( [ np.reshape(xv,(1,np.prod(xv.shape))), 
													np.reshape(yv,(1,np.prod(yv.shape))),
													np.reshape(zv,(1,np.prod(zv.shape))),
													np.ones( (1,np.prod(xv.shape))) ], axis=0 )
	xyz_camera = np.matmul(RT, xyz)

	f_x = img_shape[0] *1.2# focal distance in pixels
	f_y = f_x

	x_0 = 0
	y_0 = 0

	s = np.random.rand(1)*2*max_shear-max_shear
	K = np.asarray( [[f_x, s, x_0], [0.,f_y, y_0], [0.,0.,1.]], dtype=np.float32 )

	xy_im = np.matmul(K, xyz_camera)
	norm_row = np.tile( xy_im[-1], (3,1 ))
	xy_im = np.divide(xy_im, norm_row)

	for c in range(I.shape[-1]):
		Vq = map_coordinates( I_in[:,:,c].transpose(), xy_im[:2] + np.tile( np.asarray([[w/2],[h/2]]), (1,xy_im.shape[-1])), cval=1. )#[np.arange(0,img_shape[1]), np.arange(0,img_shape[0])] )
		I_out[:,:,c] = np.reshape(Vq, img_shape[:-1])
	return I_out,theta 
#	I_in_flat = np.reshape( I_in, 
if __name__=='__main__':
	I = cv2.imread('../datasets/MTGVS/train/1.jpg' )
	I_ap = augProjective(I, I.shape)
	cv2.imwrite('projtest.jpg',I_ap*255)
