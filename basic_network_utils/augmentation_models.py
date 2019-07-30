import sys
from basic_network_utils import network_layers

from keras.layers import Input, Lambda, Reshape
from keras.layers.pooling import MaxPooling3D
from keras.models import Model
import numpy as np

from neuron.layers import SpatialTransformer

def warp_model(img_shape, interp_mode='linear', indexing='ij'):
    n_dims = len(img_shape) - 1
    img_in = Input(img_shape, name='input_img')
    flow_in = Input(img_shape[:-1] + (n_dims,), name='input_flow')
    img_warped = SpatialTransformer(interp_mode, indexing=indexing, name='densespatialtransformer_img')([img_in, flow_in])
    
    return Model(inputs=[img_in, flow_in], outputs=img_warped, name='warp_model')

def randflow_model(img_shape,
				   model,
				   model_name='randflow_model',
				   flow_sigma=None,
				   flow_amp=None,
				   blur_sigma=5,
				   interp_mode='linear',
					indexing='xy',
				   ):
	n_dims = len(img_shape) - 1

	x_in = Input(img_shape, name='img_input_randwarp')
	#flow_placeholder = Input(img_shape[:-1] + (n_dims,), name='flow_input_placeholder')


	if n_dims == 3:
		flow = MaxPooling3D(2)(x_in)
		flow = MaxPooling3D(2)(flow)
		blur_sigma = int(np.ceil(blur_sigma / 4.))
		flow_shape = tuple([int(s/4) for s in img_shape[:-1]] + [n_dims])
	else:
		#flow = flow_placeholder
		flow = x_in
		flow_shape = img_shape[:-1] + (n_dims,)
	# random flow field
	if flow_amp is None:
		flow = network_layers.RandFlow(name='randflow', img_shape=flow_shape, blur_sigma=blur_sigma, flow_sigma=flow_sigma)(flow)
	elif flow_sigma is None:
		flow = network_layers.RandFlow_Uniform(name='randflow', img_shape=flow_shape, blur_sigma=blur_sigma, flow_amp=flow_amp)(flow)

	if n_dims == 3:
		flow = Reshape(flow_shape)(flow)
		# upsample with linear interpolation using adrian's function
		#flow = UpSampling3D(2)(flow)
		flow = Lambda(interp_upsampling)(flow)
		flow = Lambda(interp_upsampling, output_shape=img_shape[:-1] + (n_dims,))(flow)
		#flow = UpSampling3D(2)(flow)
		#flow = BlurFlow(name='blurflow', img_shape=img_shape, blur_sigma=3)(flow)
		flow = Reshape(img_shape[:-1] + (n_dims,), name='randflow_out')(flow)
#		x_warped = Dense3DSpatialTransformer(interp_method=interp_mode, name='densespatialtransformer_img')([x_in, flow])
	else:
		flow = Reshape(img_shape[:-1] + (n_dims,), name='randflow_out')(flow)
#		x_warped = Dense2DSpatialTransformer(interp_method=interp_mode, name='densespatialtransformer_img')([x_in, flow])
	x_warped = SpatialTransformer(interp_method=interp_mode, name='densespatialtransformer_img', indexing=indexing)([x_in, flow])


	if model is not None:
		model_outputs = model(x_warped)
		if not isinstance(model_outputs, list):
			model_outputs = [model_outputs]
	else:
        # a little unintuitive
		model_outputs = [x_warped, flow]
	return Model(inputs=[x_in], outputs=model_outputs, name=model_name)
