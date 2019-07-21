import sys

from cnn_utils import image_utils

import cv2
import keras.backend as K
from keras.layers import Layer
import numpy as np
import tensorflow as tf

sys.path.append('../neuron')
from neuron.layers import SpatialTransformer


def interp_upsampling(V):
	"""
	upsample a field by a factor of 2
	TODO: should switch this to use neuron.utils.interpn()
	"""
	V = tf.reshape(V, [-1] + V.get_shape().as_list()[1:])
	grid = volshape_to_ndgrid([f*2 for f in V.get_shape().as_list()[1:-1]])
	grid = [tf.cast(f, 'float32') for f in grid]
	grid = [tf.expand_dims(f/2 - f, 0) for f in grid]
	offset = tf.stack(grid, len(grid) + 1)

	V = SpatialTransformer(interp_method='linear')([V, offset])
	return V


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
		flow = RandFlow(name='randflow', img_shape=flow_shape, blur_sigma=blur_sigma, flow_sigma=flow_sigma)(flow)
	elif flow_sigma is None:
		flow = RandFlow_Uniform(name='randflow', img_shape=flow_shape, blur_sigma=blur_sigma, flow_amp=flow_amp)(flow)

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
		model_outputs = [x_warped]
	return Model(inputs=[x_in], outputs=model_outputs, name=model_name)


class Blur_Downsample(Layer):
    def __init__(self, n_chans=3, n_dims=2, do_blur=True, **kwargs):
        super(Blur_Downsample, self).__init__(**kwargs)
        scale_factor = 0.5  # we only support halving right now

        if do_blur:
            # according to scikit-image.transform.rescale documentation
            blur_sigma = (1. - scale_factor) / 2

            blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims, n_sigmas_per_side=4)
            #			if n_dims==2:
            #				blur_kernel = np.tile(np.reshape(blur_kernel, blur_kernel.shape + (1, 1)), tuple([1] * n_dims) + (n_chans, 1))
            #			else:
            blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1, 1))
            self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
        else:
            self.blur_kernel = tf.constant(np.ones([1] * n_dims + [1, 1]), dtype=tf.float32)
        self.n_dims = n_dims
        self.n_chans = n_chans

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        self.n_chans = inputs[0].get_shape().as_list()[-1]
        #		if self.n_dims == 2:
        #			blurred = tf.nn.depthwise_conv2d(inputs, self.blur_kernel,
        #											   padding='SAME', strides=[1, 2, 2, 1])
        if self.n_dims == 2:
            strides = [1, 2, 2, 1]
            conv_fn = tf.nn.conv2d
        elif self.n_dims == 3:
            strides = [1, 2, 2, 2, 1]
            conv_fn = tf.nn.conv3d

        chans_list = tf.unstack(inputs, num=self.n_chans, axis=-1)
        blurred_chans = []
        for c in range(self.n_chans):
            blurred_chan = conv_fn(tf.expand_dims(chans_list[c], axis=-1), self.blur_kernel,
                                   strides=strides, padding='SAME')
            # get rid of last dimension (size 1)
            blurred_chans.append(tf.gather(blurred_chan, 0, axis=-1))
        blurred = tf.stack(blurred_chans, axis=-1)
        return blurred

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + [int(np.ceil(s / 2)) for s in input_shape[1:-1]] + [self.n_chans])


class RandFlow_Uniform(Layer):
    def __init__(self, img_shape, blur_sigma, flow_amp, **kwargs):
        super(RandFlow_Uniform, self).__init__(**kwargs)
        n_dims = len(img_shape) - 1

        self.flow_shape = img_shape[:-1] + (n_dims,)

        blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims, n_sigmas_per_side=4)
        # TODO: make this work for 3D
        if n_dims == 2:
            blur_kernel = np.tile(np.reshape(blur_kernel, blur_kernel.shape + (1, 1)),
                                  tuple([1] * n_dims) + (n_dims, 1))
        else:
            blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1, 1))
        self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
        self.flow_amp = flow_amp
        self.n_dims = n_dims

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        if self.n_dims == 2:
            rand_flow = K.random_uniform(
                shape=tf.convert_to_tensor(
                    [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.n_dims]),
                minval=-self.flow_amp,
                maxval=self.flow_amp, dtype='float32')
            rand_flow = tf.nn.depthwise_conv2d(rand_flow, self.blur_kernel, strides=[1] * (self.n_dims + 2),
                                               padding='SAME')
        elif self.n_dims == 3:
            rand_flow = K.random_uniform(
                shape=tf.convert_to_tensor(
                    [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3], self.n_dims]),
                minval=-self.flow_amp,
                maxval=self.flow_amp, dtype='float32')

            # blur it here, then again later?
            rand_flow_list = tf.unstack(rand_flow, num=self.n_dims, axis=-1)
            flow_chans = []
            for c in range(self.n_dims):
                flow_chan = tf.nn.conv3d(tf.expand_dims(rand_flow_list[c], axis=-1), self.blur_kernel,
                                         strides=[1] * (self.n_dims + 2), padding='SAME')
                flow_chans.append(flow_chan[:, :, :, :, 0])
            rand_flow = tf.stack(flow_chans, axis=-1)
        rand_flow = tf.reshape(rand_flow, [-1] + list(self.flow_shape))
        return rand_flow

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1] + (self.n_dims,))


class BlurFlow(Layer):
    def __init__(self, img_shape, blur_sigma, **kwargs):
        super(BlurFlow, self).__init__(**kwargs)
        n_dims = len(img_shape) - 1

        self.flow_shape = tuple(img_shape[:-1]) + (n_dims,)

        blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims, n_sigmas_per_side=2)
        # TODO: make this work for 3D
        if n_dims == 2:
            blur_kernel = np.tile(np.reshape(blur_kernel, blur_kernel.shape + (1, 1)),
                                  tuple([1] * n_dims) + (n_dims, 1))
        else:
            blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1, 1))
        self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
        self.n_dims = n_dims

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        # squeeze chanenls into batch so we can do a single conv
        flow_flat = tf.transpose(inputs, [0, 4, 1, 2, 3])
        flow_flat = tf.reshape(flow_flat, [-1] + list(self.flow_shape[:-1]))
        # convolve with blurring filter
        flow_blurred = tf.nn.conv3d(tf.expand_dims(flow_flat, axis=-1), self.blur_kernel,
                                    strides=[1] * (self.n_dims + 2), padding='SAME')
        # get rid of extra channels
        flow_blurred = flow_blurred[:, :, :, :, 0]

        flow_out = tf.reshape(flow_blurred, [-1, self.n_dims] + list(self.flow_shape[:-1]))
        flow_out = tf.transpose(flow_out, [0, 2, 3, 4, 1])

        '''
        rand_flow_list = tf.unstack(inputs, num=self.n_dims, axis=-1)
        flow_chans = []
        for c in range(self.n_dims):
            flow_chan = tf.nn.conv3d(tf.expand_dims(rand_flow_list[c], axis=-1), self.blur_kernel,
                                     strides=[1] * (self.n_dims + 2), padding='SAME')
            flow_chans.append(flow_chan[:, :, :, :, 0])
        rand_flow = tf.stack(flow_chans, axis=-1)
        '''
        return flow_out


class RandFlow(Layer):
    def __init__(self, img_shape, blur_sigma, flow_sigma, normalize_max=False, **kwargs):
        super(RandFlow, self).__init__(**kwargs)
        n_dims = len(img_shape) - 1

        self.flow_shape = img_shape[:-1] + (n_dims,)

        if blur_sigma > 0:
            blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims)
            # TODO: make this work for 3D
            if n_dims == 2:
                blur_kernel = np.tile(np.reshape(blur_kernel, blur_kernel.shape + (1, 1)),
                                      tuple([1] * n_dims) + (n_dims, 1))
            else:
                blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1, 1))
            self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
        else:
            self.blur_kernel = None
        self.flow_sigma = flow_sigma
        self.normalize_max = normalize_max
        self.n_dims = n_dims
        print('Randflow dims: {}'.format(self.n_dims))

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        if self.n_dims == 2:
            rand_flow = K.random_normal(shape=tf.convert_to_tensor(
                [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.n_dims]), mean=0., stddev=1.,
                                        dtype='float32')
            rand_flow = tf.nn.depthwise_conv2d(rand_flow, self.blur_kernel, strides=[1] * (self.n_dims + 2),
                                               padding='SAME')
        elif self.n_dims == 3:
            rand_flow = K.random_normal(
                shape=tf.convert_to_tensor(
                    [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3], self.n_dims]),
                mean=0., stddev=1., dtype='float32')
            if self.blur_kernel is not None:
                rand_flow_list = tf.unstack(rand_flow, num=3, axis=-1)
                flow_chans = []
                for c in range(self.n_dims):
                    flow_chan = tf.nn.conv3d(tf.expand_dims(rand_flow_list[c], axis=-1), self.blur_kernel,
                                             strides=[1] * (self.n_dims + 2), padding='SAME')
                    flow_chans.append(flow_chan[:, :, :, :, 0])
                rand_flow = tf.stack(flow_chans, axis=-1)

        #		rand_flow = K.cast(rand_flow / tf.reduce_max(tf.abs(rand_flow)) * self.flow_sigma, dtype='float32')
        if self.normalize_max:
            rand_flow = K.cast(
                tf.add_n([rand_flow * 0, rand_flow / tf.reduce_max(tf.abs(rand_flow)) * self.flow_sigma]),
                dtype='float32')
        else:
            rand_flow = K.cast(rand_flow * self.flow_sigma, dtype='float32')
        return rand_flow

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1] + (self.n_dims,))


class DilateAndBlur(Layer):
    def __init__(self, img_shape, dilate_kernel_size, blur_sigma, flow_sigma, **kwargs):
        super(DilateAndBlur, self).__init__(**kwargs)
        n_dims = len(img_shape) - 1

        self.flow_shape = img_shape[:-1] + (n_dims,)

        dilate_kernel = np.reshape(
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)),
            (dilate_kernel_size, dilate_kernel_size, 1, 1))
        self.dilate_kernel = tf.constant(dilate_kernel, dtype=tf.float32)

        blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims)
        blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1, 1))
        blur_kernel = blur_kernel / np.max(blur_kernel)  # normalize by max instead of by sum

        self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
        self.flow_sigma = flow_sigma

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        #		errormap = inputs - tf.reduce_min(inputs)
        #		errormap = inputs / (1e-5 + tf.reduce_max(errormap))
        errormap = inputs[0]
        dilated_errormap = tf.nn.conv2d(errormap, self.dilate_kernel, strides=[1, 1, 1, 1], padding='SAME')
        blurred_errormap = tf.nn.conv2d(dilated_errormap, self.blur_kernel, strides=[1, 1, 1, 1], padding='SAME')
        blurred_errormap = K.cast(blurred_errormap / (1e-5 + tf.reduce_max(blurred_errormap)) * self.flow_sigma,
                                  dtype='float32')
        min_map = tf.tile(inputs[1][:, tf.newaxis, tf.newaxis, :],
                          tf.concat([
                              [1], tf.gather(tf.shape(blurred_errormap), [1, 2, 3])
                          ], 0))
        blurred_errormap = tf.maximum(min_map, blurred_errormap)
        return blurred_errormap
