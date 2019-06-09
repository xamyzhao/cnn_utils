import sys

import keras.initializers as keras_init
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.merge import Add, Concatenate, Multiply

import numpy as np

from cnn_utils import basic_networks
from basic_network_utils import network_layers

from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf


sys.path.append('../neuron')
from neuron.layers import SpatialTransformer

sys.path.append('../LPAT')
from networks import transform_network_utils

def transform_encoder_model(input_shapes, input_names=None,
                            latent_shape=(50,),
                            model_name='VTE_transform_encoder',
                            enc_params=None,
                            ):
    '''
    Generic encoder for a stack of inputs

    :param input_shape:
    :param latent_shape:
    :param model_name:
    :param enc_params:
    :return:
    '''
    if not isinstance(input_shapes, list):
        input_shapes = [input_shapes]

    if input_names is None:
        input_names = ['input_{}'.format(ii) for ii in range(len(input_shapes))]

    inputs = []
    for ii, input_shape in enumerate(input_shapes):
        inputs.append(Input(input_shape, name='input_{}'.format(input_names[ii])))

    inputs_stacked = Concatenate(name='concat_inputs', axis=-1)(inputs)
    input_stack_shape = inputs_stacked.get_shape().as_list()[1:]
    n_dims = len(input_stack_shape) - 1

    x_transform_enc = basic_networks.encoder(
        x=inputs_stacked,
        img_shape=input_stack_shape,
        conv_chans=enc_params['nf_enc'],
        min_h=None, min_c=None,
        n_convs_per_stage=enc_params['n_convs_per_stage'],
        use_residuals=enc_params['use_residuals'],
        use_maxpool=enc_params['use_maxpool'],
        prefix='vte'
    )

    latent_size = np.prod(latent_shape)

    if not enc_params['fully_conv']:
        # the last layer in the basic encoder will be a convolution, so we should activate after it
        x_transform_enc = LeakyReLU(0.2)(x_transform_enc)
        x_transform_enc = Flatten()(x_transform_enc)

        z_mean = Dense(latent_size, name='latent_mean',
            kernel_initializer=keras_init.RandomNormal(mean=0., stddev=0.00001))(x_transform_enc)
        z_logvar = Dense(latent_size, name='latent_logvar',
                        bias_initializer=keras_init.RandomNormal(mean=-2., stddev=1e-10),
                        kernel_initializer=keras_init.RandomNormal(mean=0., stddev=1e-10),
                    )(x_transform_enc)
    else:
        emb_shape = basic_networks.get_encoded_shape(input_stack_shape, conv_chans=enc_params['nf_enc'])
        n_chans = emb_shape[-1]

        if n_dims == 3:
            # convolve rather than Lambda since we want to set the initialization
            z_mean = Conv3D(latent_shape[-1], kernel_size=2, strides=2, padding='same',
                            kernel_initializer=keras_init.RandomNormal(mean=0., stddev=0.001))(x_transform_enc)
            z_mean = Flatten(name='latent_mean')(z_mean)

            z_logvar = Conv3D(latent_shape[-1], kernel_size=2,
                              strides=2, padding='same',
                              bias_initializer=keras_init.RandomNormal(mean=-2., stddev=1e-10),
                              kernel_initializer=keras_init.RandomNormal(mean=0., stddev=1e-10),
                              )(x_transform_enc)
            z_logvar = Flatten(name='latent_logvar')(z_logvar)
        else:
            # TODO: also convolve to latent mean and logvar for 2D?
            z_mean = Lambda(lambda x: x[:, :, :, :n_chans/2],
                            output_shape=emb_shape[:-1] + (n_chans/2,))(x_transform_enc)
            z_mean = Flatten(name='latent_mean')(z_mean)

            z_logvar = Lambda(lambda x: x[:, :, :, n_chans/2:],
                              output_shape=emb_shape[:-1] + (n_chans/2,))(x_transform_enc)
            z_logvar = Flatten(name='latent_logvar')(z_logvar)

    return Model(inputs=inputs, outputs=[z_mean, z_logvar], name=model_name)


def transformer_model(conditioning_input_shapes, conditioning_input_names=None,
                      output_shape=None,
                      model_name='CVAE_transformer',
                      transform_latent_shape=(100,),
                      transform_type=None,
                      color_transform_type=None,
                      enc_params=None,
                      condition_on_image=True,
                      n_concat_scales=3,
                      transform_activation=None, clip_output_range=None,
                      source_input_idx=None,
                      mask_by_conditioning_input_idx=None,
                      ):

    # collect conditioning inputs, and concatentate them into a stack
    if not isinstance(conditioning_input_shapes, list):
        conditioning_input_shapes = [conditioning_input_shapes]
    if conditioning_input_names is None:
        conditioning_input_names = ['cond_input_{}'.format(ii) for ii in range(len(conditioning_input_shapes))]

    conditioning_inputs = []
    for ii, input_shape in enumerate(conditioning_input_shapes):
        conditioning_inputs.append(Input(input_shape, name=conditioning_input_names[ii]))

    conditioning_input_stack = Concatenate(name='concat_cond_inputs', axis=-1)(conditioning_inputs)
    conditioning_input_shape = tuple(conditioning_input_stack.get_shape().as_list()[1:])

    n_dims = len(conditioning_input_shape) - 1
    if output_shape is None:
        output_shape = conditioning_input_shape

    # we will always give z as a flattened vector
    z_input = Input((np.prod(transform_latent_shape),), name='z_input')

    # determine what we should apply the transformation to
    if source_input_idx is None:
        # the image we want to transform is exactly the input of the conditioning branch
        x_source = conditioning_input_stack
        source_input_shape = conditioning_input_shape
    else:
        # slice conditioning input to get the single source im that we will apply the transform to
        source_input_shape = conditioning_input_shapes[source_input_idx]
        x_source = conditioning_inputs[source_input_idx]

    # assume the output is going to be the transformed source input, so it should be the same shape
    if output_shape is None:
        output_shape = source_input_shape

    if mask_by_conditioning_input_idx is None:
        x_mask = None
    else:
        print('Masking output by input {} with name {}'.format(
            mask_by_conditioning_input_idx,
            conditioning_input_names[mask_by_conditioning_input_idx]
        ))
        mask_shape = conditioning_input_shapes[mask_by_conditioning_input_idx]
        x_mask = conditioning_inputs[mask_by_conditioning_input_idx]

    if transform_type == 'flow':
        layer_prefix = 'flow'
    elif transform_type == 'color':
        layer_prefix = 'color'
    else:
        layer_prefix = 'synth'

    if condition_on_image: # assume we always condition by concat since it's better than other forms
        # simply concatenate the conditioning stack (at various scales) with the decoder volumes
        include_fullres = True

        concat_decoder_outputs_with = [None] * len(enc_params['nf_dec'])
        concat_skip_sizes = [None] * len(enc_params['nf_dec'])

        # make sure x_I is the same shape as the output, including in the channels dimension
        if not np.all(output_shape <= conditioning_input_shape):
            tile_factor = [int(round(output_shape[i] / conditioning_input_shape[i])) for i in
                           range(len(output_shape))]
            print('Tile factor: {}'.format(tile_factor))
            conditioning_input_stack = Lambda(lambda x: tf.tile(x, [1] + tile_factor), name='lambda_tile_cond_input')(conditioning_input_stack)

        # downscale the conditioning inputs by the specified number of times
        xs_downscaled = [conditioning_input_stack]
        for si in range(n_concat_scales):
            curr_x_scaled = network_layers.Blur_Downsample(
                n_chans=conditioning_input_shape[-1], n_dims=n_dims,
                do_blur=True,
                name='downsample_scale-1/{}'.format(2**(si + 1))
            )(xs_downscaled[-1])
            xs_downscaled.append(curr_x_scaled)

        if not include_fullres:
            xs_downscaled = xs_downscaled[1:]  # exclude the full-res volume

        print('Including downsampled input sizes {}'.format([x.get_shape().as_list() for x in xs_downscaled]))

        concat_decoder_outputs_with[:len(xs_downscaled)] = list(reversed(xs_downscaled))
        concat_skip_sizes[:len(xs_downscaled)] = list(reversed(
            [np.asarray(x.get_shape().as_list()[1:-1]) for x in xs_downscaled if
             x is not None]))

    else:
        # just ignore the conditioning input
        concat_decoder_outputs_with = None
        concat_skip_sizes = None


    if 'ks' not in enc_params:
        enc_params['ks'] = 3

    if not enc_params['fully_conv']:
        # determine what size to reshape the latent vector to
        reshape_encoding_to = basic_networks.get_encoded_shape(
            img_shape=conditioning_input_shape,
            conv_chans=enc_params['nf_enc'],
        )

        if np.all(reshape_encoding_to[:n_dims] > concat_skip_sizes[-1][:n_dims]):
            raise RuntimeWarning(
                'Attempting to concatenate reshaped latent vector of shape {} with downsampled input of shape {}!'.format(
                    reshape_encoding_to,
                    concat_skip_sizes[-1]
                ))

        x_enc = Dense(np.prod(reshape_encoding_to))(z_input)
    else:
        # latent representation is already in correct shape
        reshape_encoding_to = transform_latent_shape
        x_enc = z_input

    x_enc = Reshape(reshape_encoding_to)(x_enc)


    print('Decoder starting shape: {}'.format(reshape_encoding_to))

    x_transformation = basic_networks.decoder(
        x_enc, output_shape,
        encoded_shape=reshape_encoding_to,
        prefix='{}_dec'.format(layer_prefix),
        conv_chans=enc_params['nf_dec'], ks=enc_params['ks'],
        n_convs_per_stage=enc_params['n_convs_per_stage'],
        use_upsample=enc_params['use_upsample'],
        include_skips=concat_decoder_outputs_with,
        target_vol_sizes=concat_skip_sizes
    )

    if transform_activation is not None:
        x_transformation = Activation(
            transform_activation,
            name='activation_transform_{}'.format(transform_activation))(x_transformation)

        if transform_type == 'color' and 'delta' in color_transform_type and transform_activation=='tanh':
            # TODO: maybe move this logic
            # if we are learning a colro delta with a tanh, make sure to multiply it by 2
            x_transformation = Lambda(lambda x: x * 2, name='lambda_scale_tanh')(x_transformation)

    if mask_by_conditioning_input_idx is not None:
        x_transformation = Multiply(name='mult_mask_transformation')([x_transformation, x_mask])
    if transform_type is not None:
        im_out, transform_out = apply_transformation(x_source, x_transformation, 
            output_shape=source_input_shape, conditioning_input_shape=conditioning_input_shape, transform_name=transform_type,
            apply_flow_transform=transform_type=='flow',
            apply_color_transform=transform_type=='color',
            color_transform_type=color_transform_type
            )
    else:
        im_out = x_transformation

    if clip_output_range is not None:
        im_out = Lambda(lambda x: tf.clip_by_value(x, clip_output_range[0], clip_output_range[1]),
            name='lambda_clip_output_{}-{}'.format(clip_output_range[0], clip_output_range[1]))(im_out)

    if transform_type is not None:
        return Model(inputs=conditioning_inputs + [z_input], outputs=[im_out, transform_out], name=model_name)
    else:
        return Model(inputs=conditioning_inputs + [z_input], outputs=[im_out], name=model_name)


# applies a decoder to x_enc and then applies the transform to I
def apply_transformation(x_source, x_transformation,
                         output_shape,
                         conditioning_input_shape,
                         transform_name,
                         apply_flow_transform=True, apply_color_transform=False,
                         flow_indexing='xy',
                         color_transform_type='WB',
                         ):
    n_dims = len(conditioning_input_shape) - 1

    transformation_shape = x_transformation.get_shape().as_list()[1:]
    x_transformation = Reshape(transformation_shape, name='{}_dec_out'.format(transform_name))(x_transformation)

    if apply_flow_transform:
        # apply flow transform
        im_out = SpatialTransformer(name='spatial_transformer', indexing=flow_indexing)(
            [x_source, x_transformation])

    elif apply_color_transform:
        # apply color transform
        print('Applying color transform {}'.format(color_transform_type))
        if color_transform_type == 'delta':
            x_color_out = Add()([x_source, x_transformation])
        elif color_transform_type == 'mult':
            x_color_out = Multiply()([x_source, x_transformation])
        else:
            raise NotImplementedError('Only color transform types delta and mult are supported!')
        print(output_shape)
        im_out = Reshape(output_shape, name='color_transformer')(x_color_out)
    else:
        im_out = x_transformation

    return im_out, x_transformation


def cvae_trainer_wrapper(
        ae_input_shapes, ae_input_names,
        conditioning_input_shapes, conditioning_input_names,
        output_shape=None,
        model_name='transformer_trainer',
        transform_encoder_model=None, transformer_model=None,
        transform_type='flow',
        transform_latent_shape=(50,),
        include_aug_matrix=False,
        n_outputs=1,
):
    '''''''''''''''''''''
    VTE transformer train model
        - takes I, I+J as input
        - encodes I+J to z
        - condition_on_image = True means that the transform is decoded from the transform+image embedding,
                otherwise it is decoded from only the transform embedding
        - decodes latent embedding into transform and applies it
    '''''''''''''''''''''
    ae_inputs, ae_stack, conditioning_inputs, cond_stack = _collect_inputs(
        ae_input_shapes, ae_input_names,
        conditioning_input_shapes, conditioning_input_names)
    conditioning_input_shape = cond_stack.get_shape().as_list()[1:]

    inputs = ae_inputs + conditioning_inputs

    if include_aug_matrix:
        T_in = Input((3, 3), name='transform_input')
        inputs += [T_in]
        # ae_stack = SpatialTransformer(name='st_affine_stack')([ae_stack, T_in])
        # cond_stack = SpatialTransformer(name='st_affine_img')([cond_stack, T_in])
        ae_inputs = [
            SpatialTransformer(name='st_affine_{}'.format(ae_input_names[ii]))([ae_input, T_in])
            for ii, ae_input in enumerate(ae_inputs)
        ]

        conditioning_inputs = [
            SpatialTransformer(name='st_affine_{}'.format(conditioning_input_names[ii]))([cond_input, T_in])
            for ii, cond_input in enumerate(conditioning_inputs)
        ]
    # encode x_stacked into z
    z_mean, z_logvar = transform_encoder_model(ae_inputs)

    z_mean = Reshape(transform_latent_shape, name='latent_mean')(z_mean)
    z_logvar = Reshape(transform_latent_shape, name='latent_logvar')(z_logvar)

    z_sampled = Lambda(transform_network_utils.sampling, output_shape=transform_latent_shape, name='lambda_sampling')(
        [z_mean, z_logvar])

    decoder_out = transformer_model(conditioning_inputs + [z_sampled])

    if transform_type == 'flow':
        im_out, transform_out = decoder_out
        transform_shape = transform_out.get_shape().as_list()[1:]

        transform_out = Reshape(transform_shape, name='decoder_flow_out')(transform_out)
        im_out = Reshape(output_shape, name='spatial_transformer')(im_out)
    elif transform_type == 'color':
        im_out, transform_out = decoder_out

        transform_out = Reshape(output_shape, name='decoder_color_out')(transform_out)
        im_out = Reshape(output_shape, name='color_transformer')(im_out)
    else:
        im_out = decoder_out

    if transform_type is not None:
        return Model(inputs=inputs, outputs=[im_out] * n_outputs + [transform_out, z_mean, z_logvar], name=model_name)
    else:
        return Model(inputs=inputs, outputs=[im_out] * n_outputs + [z_mean, z_logvar], name=model_name)


def cvae_tester_wrapper(
        conditioning_input_shapes, conditioning_input_names,
        latent_shape,
        dec_model,

):
    # collect conditioning inputs, and concatentate them into a stack
    if not isinstance(conditioning_input_shapes, list):
        conditioning_input_shapes = [conditioning_input_shapes]
    if conditioning_input_names is None:
        conditioning_input_names = ['cond_input_{}'.format(ii) for ii in range(len(conditioning_input_shapes))]

    conditioning_inputs = []
    for ii, input_shape in enumerate(conditioning_input_shapes):
        conditioning_inputs.append(Input(input_shape, name=conditioning_input_names[ii]))

    cond_stack = Concatenate(name='concat_cond_inputs', axis=-1)(conditioning_inputs)

    z_dummy_input = Input(latent_shape, name='z_input')

    z_samp = Lambda(transform_network_utils.sampling_sigma1,
                        name='lambda_z_sampling_stdnormal'
                        )(z_dummy_input)
    y = dec_model(conditioning_inputs + [z_samp])
    return Model(inputs=conditioning_inputs + [z_dummy_input], outputs=y, name='cvae_tester_model')


def _collect_inputs(ae_input_shapes, ae_input_names,
        conditioning_input_shapes, conditioning_input_names,):

    if not isinstance(ae_input_shapes, list):
        ae_input_shapes = [ae_input_shapes]

    if ae_input_names is None:
        ae_input_names = ['input_{}'.format(ii) for ii in range(len(ae_input_names))]

    ae_inputs = []
    for ii, input_shape in enumerate(ae_input_shapes):
        ae_inputs.append(Input(input_shape, name='input_{}'.format(ae_input_names[ii])))

    ae_stack = Concatenate(name='concat_inputs', axis=-1)(ae_inputs)
    ae_stack_shape = ae_stack.get_shape().as_list()[1:]

    # collect conditioning inputs, and concatentate them into a stack
    if not isinstance(conditioning_input_shapes, list):
        conditioning_input_shapes = [conditioning_input_shapes]
    if conditioning_input_names is None:
        conditioning_input_names = ['cond_input_{}'.format(ii) for ii in range(len(conditioning_input_shapes))]

    conditioning_inputs = []
    for ii, input_shape in enumerate(conditioning_input_shapes):
        conditioning_inputs.append(Input(input_shape, name=conditioning_input_names[ii]))

    cond_stack = Concatenate(name='concat_cond_inputs', axis=-1)(conditioning_inputs)
    return ae_inputs, ae_stack, conditioning_inputs, cond_stack
