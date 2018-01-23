import numpy as np
import cv2
import os
import data_utils
import misc_utils
import PIL

def label_ims( X, Y, inverse_normalize = True, label_ims = True, clip_flow = 0, min_h = 128 ):
	batch_size = X.shape[0]
	h = X.shape[1]
	w = X.shape[2]

	if type(Y) == list and len(Y) == 1:
		Y = Y * batch_size
	elif not type(Y) == list and not type(Y) == np.ndarray:
		Y = [Y] + [' '] * (batch_size-1)

	scale_factor = min_h/float(h)
	out_im = np.zeros( (batch_size*int(min_h), int(w*scale_factor), 3), dtype=np.float32 )
	
	if len(X.shape) < 4:
		X = np.expand_dims( X, 3 )

	if X.shape[3] == 2:	# assume to be x,y flow; map to color im
		X_fullcolor = np.concatenate( [X.copy(), np.zeros( X.shape[:-1] + (1,) )], axis=3 )

		if len(Y) == 0:
			Y = [None]*batch_size
		
		for i in range(batch_size):
			X_fullcolor[i], min_flow, max_flow = flow_to_im( X[i], clip_flow = clip_flow )
			if Y[i] is not None:
				Y[i] = '{},'.format(Y[i])
			else:
				Y[i] = ''
			for c in range(len(min_flow)):
				Y[i] +='({},{})'.format( round(min_flow[c],1), round(max_flow[c],1) )
		X = X_fullcolor.copy()

	elif inverse_normalize: #np.min(X) < 0:
		X = data_utils.inverse_normalize(X)
	else:
		X = np.clip( X, 0., 1. )

	if np.max(X)<=1.0:
		X = X*255.0

	for i in range(batch_size):
		if X.shape[-1] == 1:
			curr_im =np.tile( X[i], (1,1,3))
		else:
			curr_im = X[i] 
		out_im[ i*min_h:(i+1)*min_h, :, :] = cv2.resize( curr_im, None, fx=scale_factor, fy=scale_factor )

	if len(Y) > 0:
		im_pil = PIL.Image.fromarray( out_im.astype(np.uint8) )
		draw = PIL.ImageDraw.Draw(im_pil)
		for i in range(batch_size):
			if label_ims and len(Y)>i:
	#			cv2.putText(out_im, '{}'.format(Y[i]), (5,i*min_h+5),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,0,255), 10)
				
				if type(Y[i]) == tuple or type(Y[i]) == list:
					formatted_text = ', '.join( [Y[i][j].decode('UTF-8') if type(Y[i][j])==np.unicode_ else Y[i][j] if type(Y[i][j])==str else str(round( Y[i][j],2)) if type(Y[i][j])==np.float32 else str(Y[i][j]) for j in range(len(Y[i]))  ] )
				elif type(Y[i]) == float or type(Y[i])==np.float32:
					formatted_text = str(round(Y[i],2))
				else:
					formatted_text = str(Y[i])
				font_size = min(15,int(3.*w/len(formatted_text)) )
				font=PIL.ImageFont.truetype('Ubuntu-M.ttf', font_size)
				
				draw.text( (5,i*min_h+5), '{}'.format(formatted_text), font=font, fill=(50,50,255))

		out_im = np.asarray( im_pil ) 
	return out_im

def flow_to_im( flow, clip_flow = 0 ):
	out_flow = np.zeros( flow.shape[:-1] + (3,) )

	n_chans = flow.shape[-1]

	min_flow = [None]*n_chans
	max_flow = [None]*n_chans

	for c in range(n_chans):
		curr_flow = flow[:,:,c]
		if clip_flow == 0:
			flow_vals = np.sort( curr_flow.flatten(), axis=None )
			min_flow[c] = flow_vals[ int(0.05*len(flow_vals)) ]
			max_flow[c] = flow_vals[ int(0.95*len(flow_vals)) ]
		else:
			min_flow[c] = -clip_flow
			max_flow[c] = clip_flow

		curr_flow = ( curr_flow - min_flow[c] ) * 1./ ( max_flow[c]-min_flow[c] )
		curr_flow = np.clip( curr_flow, 0., 1. )
		out_flow[:,:,c] = curr_flow * 255
		#flow = np.concatenate( [ flow, np.zeros( flow.shape[:-1]+(1,) ) ], axis=-1 ) * 255
	return out_flow, min_flow, max_flow

def xy_flow_to_im_cmap( flow ):
	# assume flow is h x w x 2
	n_vals = 256
	cm = misc_utils.make_cmap_rainbow( n_vals )

	flow_mag = np.sqrt( flow[:,:,0]**2 + flow[:,:,1]**2 )
	flow_mag /= np.max(flow_mag )
	flow_angle = np.arctan2( flow[:,:,1], flow[:,:,0] )

	flow_angle_binned = np.digitize( flow_angle, np.linspace(0, 255, n_vals+1))

	flow_im = cm[flow_angle_binned]

	flow_im_hsv = cv2.cvtColor( flow_im, cv2.COLOR_RGB2HSV )
	flow_im_hsv[:,:,1] = flow_mag

	flow_im = cv2.cvtColor( flow_im_hsv, cv2.COLOR_HSV2RGB)
	return flow_im
	

#def overlay_labels_on_im( im, labels):
	# assume labels is batch_size x h x w x n_labels
