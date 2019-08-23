from keras.layers import Layer
from keras.layers import BatchNormalization, Input, Flatten, Dense, Reshape, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.layers.convolutional import Conv3D, UpSampling3D, Conv3DTranspose, Cropping3D, ZeroPadding3D, MaxPooling3D
from keras.layers.merge import Add, Concatenate, Multiply
from keras.layers.pooling import MaxPooling2D
from keras import regularizers, initializers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K
import numpy as np
import tensorflow as tf

def autoencoder(img_shape, latent_dim=10,
                conv_chans = None,
                n_convs_per_stage=1,
                fully_conv=False,
                name_prefix = '',
                activation=None,
                return_encoding=False,
    ):
    x_input = Input(img_shape)

    x_enc = encoder(x_input, img_shape, conv_chans = conv_chans,
                n_convs_per_stage = n_convs_per_stage,
                prefix = 'autoencoder_enc')
    preflatten_shape = get_encoded_shape(img_shape=img_shape, conv_chans=conv_chans)
    print('Autoencoder preflatten shape {}'.format(preflatten_shape))

    if fully_conv:
        x_enc = Flatten()(x_enc)
        x_enc = Dense(latent_dim, name='autoencoder_latent')(x_enc)
        x = Dense(np.prod(preflatten_shape))(x_enc)
    else:
        x = Reshape(preflatten_shape, name='autoencoder_latent')(x_enc)

    x = Reshape(preflatten_shape)(x)
    y = decoder(x, img_shape, conv_chans=conv_chans,
                encoded_shape=preflatten_shape,
                n_convs_per_stage=n_convs_per_stage,
                prefix = 'autoencoder_dec',
                min_h = preflatten_shape[0])
    if activation is not None:
        y = Activation(activation, name='activation_{}'.format(activation))(y)

    if return_encoding:
        return Model(inputs=[x_input], outputs=[x_enc, y],
                     name=name_prefix + '_autoencoder')
    else:
        return Model(inputs=[x_input], outputs=[y],
                     name=name_prefix + '_autoencoder')


def myConv(nf, n_dims, prefix=None, suffix=None, ks=3, strides=1, initializer=None):
    if initializer is None:
        initializer = 'glorot_uniform' # keras default for conv kernels

    # wrapper for 2D and 3D conv
    if n_dims == 2:
        if not isinstance(strides, tuple):
            strides = (strides, strides)
        return Conv2D(nf, kernel_size=ks, padding='same', strides=strides, kernel_initializer=initializer,
                      name='_'.join([
                          str(part) for part in [prefix, 'conv2D', suffix]  # include prefix and suffix if they exist
                          if part is not None and len(str(part)) > 0]))
    elif n_dims == 3:
        if not isinstance(strides, tuple):
            strides = (strides, strides, strides)
        return Conv3D(nf, kernel_size=ks, padding='same', strides=strides, kernel_initializer=initializer,
                      name='_'.join([
                          str(part) for part in [prefix, 'conv3D', suffix]  # include prefix and suffix if they exist
                          if part is not None and len(str(part)) > 0]))
    else:
        print('N dims {} is not supported!'.format(n_dims))
        sys.exit()

def myPool(n_dims, prefix=None, suffix=None):
    if n_dims == 2:
        return MaxPooling2D(padding='same',
                            name='_'.join([
                                str(part) for part in [prefix, 'maxpool2D', suffix]  # include prefix and suffix if they exist
                                if part is not None and len(str(part)) > 0]))
    elif n_dims == 3:
        return MaxPooling3D(padding='same',
                            name='_'.join([
                                str(part) for part in [prefix, 'maxpool3D', suffix]  # include prefix and suffix if they exist
                                if part is not None and len(str(part)) > 0]))
'''''''''
Basic encoder/decoders
- chans per stage specified
- additional conv with no activation at end to desired shape
'''''''''

def encoder(x, img_shape,
            conv_chans=None,
            n_convs_per_stage=1,
            min_h=5, min_c=None,
            prefix='',
            ks=3,
            return_skips=False, use_residuals=False, use_maxpool=False, use_batchnorm=False, initializer=None):
    skip_layers = []
    concat_skip_sizes = []
    n_dims = len(img_shape) - 1  # assume img_shape includes spatial dims, followed by channels

    if conv_chans is None:
        n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
        conv_chans = [min_c * 2] * (n_convs - 1) + [min_c]
    elif not type(conv_chans) == list:
        n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
        conv_chans = [conv_chans] * (n_convs - 1) + [min_c]
    else:
        n_convs = len(conv_chans)

    if isinstance(ks, list):
        assert len(ks) == (n_convs + 1)  # specify for each conv, as well as the last one
    else:
        ks = [ks] * (n_convs + 1)


    for i in range(len(conv_chans)):
        #if n_convs_per_stage is not None and n_convs_per_stage > 1 or use_maxpool and n_convs_per_stage is not None:
        for ci in range(n_convs_per_stage):
            x = myConv(nf=conv_chans[i], ks=ks[i], strides=1, n_dims=n_dims, initializer=initializer,
                       prefix='{}_enc'.format(prefix),
                       suffix='{}_{}'.format(i, ci + 1))(x)

            if ci == 0 and use_residuals:
                residual_input = x
            elif ci == n_convs_per_stage - 1 and use_residuals:
                x = Add(name='{}_enc_{}_add_residual'.format(prefix, i))([residual_input, x])

            if use_batchnorm:
                x = BatchNormalization()(x)
            x = LeakyReLU(0.2, name='{}_enc_leakyrelu_{}_{}'.format(prefix, i, ci + 1))(x)

        if return_skips:
            skip_layers.append(x)
            concat_skip_sizes.append(np.asarray(x.get_shape().as_list()[1:-1]))

        if use_maxpool and i < len(conv_chans) - 1:
            # changed 5/30/19, don't pool after our last conv
            x = myPool(n_dims=n_dims, prefix=prefix, suffix=i)(x)
        else:
            x = myConv(conv_chans[i], ks=ks[i], strides=2, n_dims=n_dims,
                       prefix='{}_enc'.format(prefix), suffix=i)(x)

            # don't activate right after a maxpool, it makes no sense
            if i < len(conv_chans) - 1:  # no activation on last convolution
                x = LeakyReLU(0.2, name='{}_enc_leakyrelu_{}'.format(prefix, i))(x)

    if min_c is not None and min_c > 0:
        # if the last number of channels is specified, convolve to that
        if n_convs_per_stage is not None and n_convs_per_stage > 1:
            for ci in range(n_convs_per_stage):
                # TODO: we might not have enough ks for this
                x = myConv(min_c, ks=ks[-1], n_dims=n_dims, strides=1,
                           prefix='{}_enc'.format(prefix), suffix='last_{}'.format(ci + 1))(x)

                if ci == 0 and use_residuals:
                    residual_input = x
                elif ci == n_convs_per_stage - 1 and use_residuals:
                    x = Add(name='{}_enc_{}_add_residual'.format(prefix, 'last'))([residual_input, x])
                x = LeakyReLU(0.2, name='{}_enc_leakyrelu_last'.format(prefix))(x)

        x = myConv(min_c, ks=ks[-1], strides=1, n_dims=n_dims,
                   prefix='{}_enc'.format(prefix),
                   suffix='_last')(x)

        if return_skips:
            skip_layers.append(x)
            concat_skip_sizes.append(np.asarray(x.get_shape().as_list()[1:-1]))

    if return_skips:
        return x, skip_layers, concat_skip_sizes
    else:
        return x


def encoder_model(img_shape,
                  dense_size=None,
                  activation=None,
                conv_chans = None,
                n_convs_per_stage = 1,
                min_h = 5, min_c = None,
                prefix = '',
                ks = 3,
             return_skips=False, use_residuals=False, use_maxpool=False):
    x = Input(img_shape, name='{}_enc_input'.format(prefix))
    y = encoder(x, img_shape=img_shape,
                conv_chans=conv_chans,
                n_convs_per_stage=n_convs_per_stage,
                prefix=prefix, ks=ks,
                return_skips=return_skips,
                use_residuals=use_residuals,
                use_maxpool=use_maxpool
                )
    if dense_size is not None:
        y = Flatten()(y)
        y = Dense(dense_size)(y)
    if activation is not None:
        y = Activation(activation)(y)
    return Model(inputs=x, outputs=y, name='{}_encoder_model'.format(prefix))


'''''''''
Basic encoder/decoders
- chans per stage specified
- additional conv with no activation at end to desired shape
'''''''''
def encoder3D(x, img_shape,
            conv_chans=None,
            n_convs_per_stage=1,
            min_h=5, min_c=None,
            prefix='vte',
            ks=3,
            return_skips=False, use_residuals=False, use_maxpool=False,
            max_time_downsample=None):
    skip_layers = []
    concat_skip_sizes = []

    if max_time_downsample is None:
        # do not attempt to downsample beyond 1
        max_time_downsample = int(np.floor(np.log2(img_shape[-2]))) - 1
        print('Max downsamples in time: {}'.format(max_time_downsample))

    if conv_chans is None:
        n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
        conv_chans = [min_c * 2] * (n_convs - 1) + [min_c]
    elif not type(conv_chans) == list:
        n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
        conv_chans = [conv_chans] * (n_convs - 1) + [min_c]

    for i in range(len(conv_chans)):
        if n_convs_per_stage is not None and n_convs_per_stage > 1 or use_maxpool and n_convs_per_stage is not None:
            for ci in range(n_convs_per_stage):
                x = Conv3D(conv_chans[i], kernel_size=ks, padding='same',
                           name='{}_enc_conv3D_{}_{}'.format(prefix, i, ci + 1))(x)
                if ci == 0 and use_residuals:
                    residual_input = x
                elif ci == n_convs_per_stage - 1 and use_residuals:
                    x = Add(name='{}_enc_{}_add_residual'.format(prefix, i))([residual_input, x])

                x = LeakyReLU(0.2, name='{}_enc_leakyrelu_{}_{}'.format(prefix, i, ci + 1))(x)

        if return_skips:
            skip_layers.append(x)
            concat_skip_sizes.append(x.get_shape().as_list()[1:-1])

        # only downsample if we are below the max number of downsamples in time
        if i < max_time_downsample:
            strides = (2, 2, 2)
        else:
            strides = (2, 2, 1)

        if use_maxpool:
            x = MaxPooling3D(pool_size=strides,
                             name='{}_enc_maxpool_{}'.format(prefix, i))(x)
        else:
            x = Conv3D(conv_chans[i], kernel_size=ks, strides=strides, padding='same',
                       name='{}_enc_conv3D_{}'.format(prefix, i))(x)

        if i < len(conv_chans) - 1:  # no activation on last convolution
            x = LeakyReLU(0.2, name='{}_enc_leakyrelu_{}'.format(prefix, i))(x)

    if min_c is not None and min_c > 0:
        if n_convs_per_stage is not None and n_convs_per_stage > 1:
            for ci in range(n_convs_per_stage):
                x = Conv3D(min_c, kernel_size=ks, padding='same',
                           name='{}_enc_conv3D_last_{}'.format(prefix, ci + 1))(x)
                if ci == 0 and use_residuals:
                    residual_input = x
                elif ci == n_convs_per_stage - 1 and use_residuals:
                    x = Add(name='{}_enc_{}_add_residual'.format(prefix, 'last'))([residual_input, x])
                x = LeakyReLU(0.2, name='{}_enc_leakyrelu_last'.format(prefix))(x)
        x = Conv3D(min_c, kernel_size=ks, strides=(1, 1, 1), padding='same',
                   name='{}_enc_conv3D_last'.format(prefix))(x)
        if return_skips:
            skip_layers.append(x)
            concat_skip_sizes.append(x.get_shape().as_list()[1:-1])

    if return_skips:
        return x, skip_layers, concat_skip_sizes
    else:
        return x

def myConvTranspose(nf, n_dims, prefix=None, suffix=None, ks=3, strides=1):
    # wrapper for 2D and 3D conv
    if n_dims == 2:
        if not isinstance(strides, tuple):
            strides = (strides, strides)
        return Conv2DTranspose(nf, kernel_size=ks, padding='same', strides=strides,
                      name='_'.join([
                          str(part) for part in [prefix, 'conv2Dtrans', suffix]  # include prefix and suffix if they exist
                          if part is not None and len(str(part)) > 0]))
    elif n_dims == 3:
        if not isinstance(strides, tuple):
            strides = (strides, strides, strides)
        return Conv3DTranspose(nf, kernel_size=ks, padding='same', strides=strides,
                      name='_'.join([
                          str(part) for part in [prefix, 'conv3Dtrans', suffix]  # include prefix and suffix if they exist
                          if part is not None and len(str(part)) > 0]))

def myUpsample(n_dims, size=2, prefix=None, suffix=None):
    if n_dims == 2:
        if not isinstance(size, tuple):
            size = (size, size)

        return UpSampling2D(size=size,
                            name='_'.join([
                                str(part) for part in [prefix, 'upsamp2D', suffix]  # include prefix and suffix if they exist
                                if part is not None and len(str(part)) > 0]))
    elif n_dims == 3:
        if not isinstance(size, tuple):
            size = (size, size, size)

        return UpSampling3D(size=size,
                            name='_'.join([
                                str(part) for part in [prefix, 'upsamp3D', suffix]  # include prefix and suffix if they exist
                                if part is not None and len(str(part)) > 0]))


def decoder(x,
            output_shape,
            encoded_shape,
            conv_chans=None,
            min_h=5, min_c=4,
            prefix='vte_dec',
            n_convs_per_stage=1,
            include_dropout=False,
            ks=3,
            include_skips=None,
            use_residuals=False,
            use_upsample=False,
            use_batchnorm=False,
            target_vol_sizes=None,
            n_samples=1,
            ):
    n_dims = len(output_shape) - 1
    if conv_chans is None:
        n_convs = int(np.floor(np.log2(output_shape[0] / min_h)))
        conv_chans = [min_c * 2] * n_convs
    elif not type(conv_chans) == list:
        n_convs = int(np.floor(np.log2(output_shape[0] / min_h)))
        conv_chans = [conv_chans] * n_convs
    elif type(conv_chans) == list:
        n_convs = len(conv_chans)

    if isinstance(ks, list):
        assert len(ks) == (n_convs + 1)  # specify for each conv, as well as the last one
    else:
        ks = [ks] * (n_convs + 1)

    print('Decoding with conv filters {}'.format(conv_chans))
    # compute default sizes that we want on the way up, mainly in case we have more convs than stages
    # and we upsample past the output size
    if n_dims == 2:
        # just upsample by a factor of 2 and then crop the final volume to the desired volume
        default_target_vol_sizes = np.asarray(
            [(int(encoded_shape[0] * 2. ** (i + 1)), int(encoded_shape[1] * 2. ** (i + 1)))
             for i in range(n_convs - 1)] + [output_shape[:2]])
    else:
        print(output_shape)
        print(encoded_shape)
        # just upsample by a factor of 2 and then crop the final volume to the desired volume
        default_target_vol_sizes = np.asarray(
            [(
                min(output_shape[0], int(encoded_shape[0] * 2. ** (i + 1))),
                min(output_shape[1], int(encoded_shape[1] * 2. ** (i + 1))),
                min(output_shape[2], int(encoded_shape[2] * 2. ** (i + 1))))
            for i in range(n_convs - 1)] + [output_shape[:3]])

    # automatically stop when we reach the desired image shape
    for vi, vs in enumerate(default_target_vol_sizes):
        if np.all(vs >= output_shape[:-1]):
            default_target_vol_sizes[vi] = output_shape[:-1]
    print('Automatically computed target output sizes: {}'.format(default_target_vol_sizes))

    if target_vol_sizes is None:
        target_vol_sizes = default_target_vol_sizes
    else:
        print('Target concat vols to match shapes to: {}'.format(target_vol_sizes))

        # TODO: check that this logic makes sense for more convs
        # fill in any Nones that we might have in our target_vol_sizes
        filled_target_vol_sizes = [None] * len(target_vol_sizes)
        for i in range(n_convs):
            if i < len(target_vol_sizes) and target_vol_sizes[i] is not None:
                filled_target_vol_sizes[i] = target_vol_sizes[i]
        target_vol_sizes = filled_target_vol_sizes

    if include_skips is not None:
        print('Concatentating padded/cropped shapes {} with skips {}'.format(target_vol_sizes, include_skips))

    curr_shape = np.asarray(encoded_shape[:n_dims])
    for i in range(n_convs):
        print(target_vol_sizes[i])
        if i < len(target_vol_sizes) and target_vol_sizes[i] is not None:
            x = _pad_or_crop_to_shape(x, curr_shape, target_vol_sizes[i])
            curr_shape = np.asarray(target_vol_sizes[i])  # we will upsample first thing next stage

        # if we want to concatenate with another volume (e.g. from encoder, or a downsampled input)...
        if include_skips is not None and i < len(include_skips) and include_skips[i] is not None:
            x_shape = x.get_shape().as_list()
            skip_shape = include_skips[i].get_shape().as_list()

            print('Attempting to concatenate current layer {} with previous skip connection {}'.format(x_shape, skip_shape))
            # input size might not match in time dimension, so just tile it
            if n_samples > 1:
                tile_factor = [1] + [n_samples] + [1] * (len(x_shape)-1)
                print('Tiling by {}'.format(tile_factor))
                print(target_vol_sizes[i])
                skip = Lambda(lambda y: K.expand_dims(y, axis=1))(include_skips[i])
                skip = Lambda(lambda y:tf.tile(y, tile_factor), name='{}_lambdatilesamples_{}'.format(prefix,i),
                    output_shape=[n_samples] + skip_shape[1:]
                    )(skip)
                skip = Lambda(lambda y:tf.reshape(y, [-1] + skip_shape[1:]), output_shape=skip_shape[1:])(skip)
            else:
                skip = include_skips[i]

            x = Concatenate(axis=-1, name='{}_concatskip_{}'.format(prefix, i))([x, skip])

        for ci in range(n_convs_per_stage):
            x = myConv(conv_chans[i],
                       ks=ks[i],
                       strides=1,
                       n_dims=n_dims,
                       prefix=prefix,
                       suffix='{}_{}'.format(i, ci + 1))(x)
            if use_batchnorm: # TODO: check to see if this should go before the residual
                x = BatchNormalization()(x)
            # if we want residuals, store them here
            if ci == 0 and use_residuals:
                residual_input = x
            elif ci == n_convs_per_stage - 1 and use_residuals:
                x = Add(name='{}_{}_add_residual'.format(prefix, i))([residual_input, x])
            x = LeakyReLU(0.2,
                          name='{}_leakyrelu_{}_{}'.format(prefix, i, ci + 1))(x)

        if include_dropout and i < 2:
            x = Dropout(0.3)(x)

        # if we are not at the output resolution yet, upsample or do a transposed convolution
        if not np.all(curr_shape == output_shape[:len(curr_shape)]):
            if not use_upsample:
                # if we have convolutional filters left at the end, just apply them at full resolution
                x = myConvTranspose(conv_chans[i], n_dims=n_dims,
                                    ks=ks[i], strides=2,
                                    prefix=prefix, suffix=i,
                                    )(x)
                if use_batchnorm:
                    x = BatchNormalization()(x)
                x = LeakyReLU(0.2, name='{}_leakyrelu_{}'.format(prefix, i))(x)  # changed 5/15/2018, will break old models
            else:
                x = myUpsample(size=2, n_dims=n_dims, prefix=prefix, suffix=i)(x)
            curr_shape *= 2

    # last stage of convolutions, no more upsampling
    x = myConv(output_shape[-1], ks=ks[-1], n_dims=n_dims,
               strides=1,
               prefix=prefix,
               suffix='final',
               )(x)

    return x


def get_encoded_shape( img_shape, min_c = None, conv_chans = None, n_convs = None):
    if n_convs is None:
        n_convs = len(conv_chans)
        min_c = conv_chans[-1]
    encoded_shape = tuple([int(np.ceil(s/2. ** n_convs)) for s in img_shape[:-1]] + [min_c])
    #encoded_shape = (int(np.ceil(img_shape[0]/2.**n_convs)), int(np.ceil(img_shape[1]/2.**n_convs)), min_c)
    print('Encoded shape for img {} with {} convs is {}'.format(img_shape, n_convs, encoded_shape))
    return encoded_shape


def _pad_or_crop_to_shape(x, in_shape, tgt_shape):
    if len(in_shape) == 2:
        '''
        in_shape, tgt_shape are both 2x1 numpy arrays
        '''
        in_shape = np.asarray(in_shape)
        tgt_shape = np.asarray(tgt_shape)
        print('Padding input from {} to {}'.format(in_shape, tgt_shape))
        im_diff = in_shape - tgt_shape
        if im_diff[0] < 0:
            pad_amt = (int(np.ceil(abs(im_diff[0])/2.0)), int(np.floor(abs(im_diff[0])/2.0)))
            x = ZeroPadding2D( (pad_amt, (0,0)) )(x)
        if im_diff[1] < 0:
            pad_amt = (int(np.ceil(abs(im_diff[1])/2.0)), int(np.floor(abs(im_diff[1])/2.0)))
            x = ZeroPadding2D( ((0,0), pad_amt) )(x)

        if im_diff[0] > 0:
            crop_amt = (int(np.ceil(im_diff[0]/2.0)), int(np.floor(im_diff[0]/2.0)))
            x = Cropping2D( (crop_amt, (0,0)) )(x)
        if im_diff[1] > 0:
            crop_amt = (int(np.ceil(im_diff[1]/2.0)), int(np.floor(im_diff[1]/2.0)))
            x = Cropping2D( ((0,0),crop_amt) )(x)
        return x
    else:
        return _pad_or_crop_to_shape_3D(x, in_shape, tgt_shape)

def _pad_or_crop_to_shape_3D(x, in_shape, tgt_shape):
    '''
    in_shape, tgt_shape are both 2x1 numpy arrays
    '''
    im_diff = np.asarray(in_shape[:3]) - np.asarray(tgt_shape[:3])
    print(im_diff)
    if im_diff[0] < 0:
        pad_amt = (int(np.ceil(abs(im_diff[0])/2.0)), int(np.floor(abs(im_diff[0])/2.0)))
        x = ZeroPadding3D((
                pad_amt,
                (0,0),
                (0,0)
        ))(x)
    if im_diff[1] < 0:
        pad_amt = (int(np.ceil(abs(im_diff[1])/2.0)), int(np.floor(abs(im_diff[1])/2.0)))
        x = ZeroPadding3D(((0,0), pad_amt, (0,0)) )(x)
    if im_diff[2] < 0:
        pad_amt = (int(np.ceil(abs(im_diff[2])/2.0)), int(np.floor(abs(im_diff[2])/2.0)))
        x = ZeroPadding3D(((0,0), (0,0), pad_amt))(x)

    if im_diff[0] > 0:
        crop_amt = (int(np.ceil(im_diff[0]/2.0)), int(np.floor(im_diff[0]/2.0)))
        x = Cropping3D((crop_amt, (0,0), (0,0)))(x)
    if im_diff[1] > 0:
        crop_amt = (int(np.ceil(im_diff[1]/2.0)), int(np.floor(im_diff[1]/2.0)))
        x = Cropping3D(((0,0), crop_amt, (0,0)))(x)
    if im_diff[2] > 0:
        crop_amt = (int(np.ceil(im_diff[2]/2.0)), int(np.floor(im_diff[2]/2.0)))
        x = Cropping3D(((0,0), (0,0), crop_amt))(x)
    return x

# wraps a unet2D and unet3D as a model for ease of visualization
def unet_model(unet_fn, unet_input_shape=None, model_name=None, **kwargs):
    if unet_input_shape is None:
        unet_input_shape = kwargs['input_shape']
    if model_name is None:
        model_name = kwargs['layer_prefix']
    unet_input = Input(unet_input_shape, name='unet_input')
    out = unet_fn(unet_input, **kwargs)
    return Model(inputs=unet_input, outputs=out, name=model_name)

def unet2D(x_in,
           input_shape, out_im_chans,
           nf_enc=[64, 64, 128, 128, 256, 256, 512],
           nf_dec=None,
           regularizer=None, initializer=None, layer_prefix='unet',
           n_convs_per_stage=1,
           use_residuals=False,
           use_maxpool=False,
           concat_at_stages=None,
           do_last_conv=True,
           ks=3,
           ):

    reg_params = {}
    if regularizer == 'l1':
        reg = regularizers.l1(1e-6)
    else:
        reg = None

    if initializer == 'zeros':
        reg_params['kernel_initializer'] = initializers.Zeros()

    x = x_in
    encodings = []
    for i in range(len(nf_enc)):
        if not use_maxpool and i > 0:
            x = LeakyReLU(0.2)(x)

        for j in range(n_convs_per_stage):
            if nf_enc[i] is not None:  # in case we dont want to convolve at the first resolution
                x = Conv2D(nf_enc[i],
                           kernel_regularizer=reg, kernel_size=ks,
                           strides=(1, 1), padding='same',
                           name='{}_enc_conv2D_{}_{}'.format(layer_prefix, i, j + 1))(x)

            if concat_at_stages and concat_at_stages[i] is not None:
                x = Concatenate(axis=-1)([x, concat_at_stages[i]])

            if j == 0 and use_residuals:
                residual_input = x
            elif j == n_convs_per_stage - 1 and use_residuals:
                x = Add()([residual_input, x])
            x = LeakyReLU(0.2)(x)

        if i < len(nf_enc) - 1:
            encodings.append(x)
            if use_maxpool:
                x = MaxPooling2D(pool_size=(2, 2), padding='same',
                                 name='{}_enc_maxpool_{}'.format(layer_prefix, i))(x)
            else:
                x = Conv2D(nf_enc[i], kernel_size=ks, strides=(2, 2), padding='same',
                           name='{}_enc_conv2D_{}'.format(layer_prefix, i))(x)

    print('Encodings to concat later: {}'.format(encodings))
    if nf_dec is None:
        nf_dec = list(reversed(nf_enc[1:]))
    print('Decoder channels: {}'.format(nf_dec))

    for i in range(len(nf_dec)):
        curr_shape = x.get_shape().as_list()[1:-1]

        # if we're not at full resolution, keep upsampling
        if np.any(list(curr_shape[:2]) < list(input_shape[:2])):
            x = UpSampling2D(size=(2, 2), name='{}_dec_upsamp_{}'.format(layer_prefix, i))(x)

        # if we still have things to concatenate, do that
        if (i + 1) <= len(encodings):
            curr_shape = x.get_shape().as_list()[1:-1]
            concat_with_shape = encodings[-i - 1].get_shape().as_list()[1:-1]
            x = _pad_or_crop_to_shape(x, curr_shape, concat_with_shape)
            x = Concatenate()([x, encodings[-i - 1]])

        residual_input = x

        for j in range(n_convs_per_stage):
            x = Conv2D(nf_dec[i],
                       kernel_regularizer=reg,
                       kernel_size=ks, strides=(1, 1), padding='same',
                       name='{}_dec_conv2D_{}_{}'.format(layer_prefix, i, j))(x)
            if j == 0 and use_residuals:
                residual_input = x
            elif j == n_convs_per_stage - 1 and use_residuals:
                x = Add()([residual_input, x])
            x = LeakyReLU(0.2)(x)

    #x = Concatenate()([x, encodings[0]])
    '''
    for j in range(n_convs_per_stage - 1):
        x = Conv2D(out_im_chans,
                   kernel_regularizer=reg,
                   kernel_size=ks, strides=(1, 1), padding='same',
                   name='{}_dec_conv2D_last_{}'.format(layer_prefix, j))(x)
        x = LeakyReLU(0.2)(x)
    '''
    if do_last_conv:
        y = Conv2D(out_im_chans, kernel_size=1, padding='same', kernel_regularizer=reg,
               name='{}_dec_conv2D_final'.format(layer_prefix))(x)  # add your own activation after this model
    else:
        y = x

    # add your own activation after this model
    return y


def segnet2D(x_in,
           img_shape, out_im_chans,
           nf_enc=[64, 64, 128, 128, 256, 256, 512],
           nf_dec=None,
           regularizer=None, initializer=None, layer_prefix='segnet',
           n_convs_per_stage=1,
           include_residual=False,
           concat_at_stages=None):
    ks = 3

    encoding_im_sizes = np.asarray([(int(np.ceil(img_shape[0]/2.0**i)), int(np.ceil(img_shape[1]/2.0**i))) \
                                    for i in range(0, len(nf_enc) + 1)])


    reg_params = {}
    if regularizer=='l1':
        reg = regularizers.l1(1e-6)
    else:
        reg = None

    if initializer=='zeros':
        reg_params['kernel_initializer'] = initializers.Zeros()

    x = x_in
    # start with the input channels
    encoder_pool_idxs = []

    for i in range(len(nf_enc)):
        for j in range(n_convs_per_stage):
            x = Conv2D(nf_enc[i],
                       kernel_regularizer=reg, kernel_size=ks,
                       strides=(1,1), padding='same',
                       name='{}_enc_conv2D_{}_{}'.format(layer_prefix, i, j+1))(x)

            if concat_at_stages and concat_at_stages[i] is not None:
                x = Concatenate(axis=-1)([x, concat_at_stages[i]])

            if j==0 and include_residual:
                residual_input = x
            elif j==n_convs_per_stage-1 and include_residual:
                x = Add()([residual_input, x])
            x = LeakyReLU(0.2)(x)

        x, pool_idxs = MaxPoolingWithArgmax2D(pool_size=(ks, ks), padding='same', name='{}_enc_maxpool_{}'.format(layer_prefix, i))(x)
        encoder_pool_idxs.append(pool_idxs)


    if nf_dec is None:
        nf_dec = list(reversed(nf_enc[1:]))

    decoding_im_sizes = [ encoding_im_sizes[-1]*2 ]
    for i in range(len(nf_dec)):
        x = MaxUnpooling2D()([x, encoder_pool_idxs[-1-i]])
        x = _pad_or_crop_to_shape(x, decoding_im_sizes[-1], encoding_im_sizes[-i-2] )


        decoding_im_sizes.append( encoding_im_sizes[-i-2] * 2 ) # the next deconv layer will produce this image height

        residual_input = x

        for j in range(n_convs_per_stage):
            x = Conv2D(nf_dec[i],
                       kernel_regularizer=reg,
                       kernel_size=ks, strides=(1,1), padding='same',
                       name='{}_dec_conv2D_{}_{}'.format(layer_prefix, i, j))(x)
            if j==0 and include_residual:
                residual_input = x
            elif j==n_convs_per_stage-1 and include_residual:
                x = Add()([residual_input, x])
            x = LeakyReLU(0.2)(x)

        if i < len(nf_dec) - 1:
            # an extra conv compared to unet, so that the unpool op gets the right number of filters
            x = Conv2D(nf_dec[i + 1],
                       kernel_regularizer=reg,
                       kernel_size=ks, strides=(1, 1), padding='same',
                       name='{}_dec_conv2D_{}_extra'.format(layer_prefix, i))(x)
            x = LeakyReLU(0.2)(x)

    y = Conv2D( out_im_chans, kernel_size=1, padding='same', kernel_regularizer=reg,
            name='{}_dec_conv2D_last_last'.format(layer_prefix))(x)		# add your own activation after this model
    # add your own activation after this model
    return y

def unet3D(x_in,
           img_shape, out_im_chans,
           nf_enc=[64, 64, 128, 128, 256, 256, 512],
           nf_dec=None,
           regularizer=None, initializer=None, layer_prefix='unet',
           n_convs_per_stage=1,
           include_residual=False, use_maxpool=True,
           max_time_downsample=None,
           n_tasks=1,
           use_dropout=False,
           do_unpool=False,
            do_last_conv=True,
        ):
    ks = 3
    if max_time_downsample is None:
        max_time_downsample = len(nf_enc)  # downsample in time all the way down

        encoding_im_sizes = np.asarray([(
                    int(np.ceil(img_shape[0] / 2.0 ** i)),
                    int(np.ceil(img_shape[1] / 2.0 ** i)),
                    int(np.ceil(img_shape[2] / 2.0 ** i)),
                ) for i in range(0, len(nf_enc) + 1)])
    else:
        encoding_im_sizes = np.asarray([(
                    int(np.ceil(img_shape[0] / 2.0 ** i)),
                    int(np.ceil(img_shape[1] / 2.0 ** i)),
                    max(int(np.ceil(img_shape[2] / 2.0 ** (max_time_downsample))), int(np.ceil(img_shape[2] / 2.0 ** i))),
                ) for i in range(0, len(nf_enc) + 1)])

    reg_params = {}
    if regularizer == 'l1':
        reg = regularizers.l1(1e-6)
    else:
        reg = None

    if initializer == 'zeros':
        reg_params['kernel_initializer'] = initializers.Zeros()

    x = x_in

    encodings = []
    encoding_im_sizes = []
    for i in range(len(nf_enc)):
        if not use_maxpool and i > 0:
            x = LeakyReLU(0.2)(x)

        for j in range(n_convs_per_stage):
            if nf_enc[i] is not None:  # in case we dont want to convovle at max resolution
                x = Conv3D(
                    nf_enc[i],
                    kernel_regularizer=reg, kernel_size=ks,
                    strides=(1, 1, 1), padding='same',
                    name='{}_enc_conv3D_{}_{}'.format(layer_prefix, i, j + 1))(x)
            #if use_dropout:
            #	x = Dropout(0.2)(x)

            if j == 0 and include_residual:
                residual_input = x
            elif j == n_convs_per_stage - 1 and include_residual:
                x = Add()([residual_input, x])

            x = LeakyReLU(0.2)(x)

        encodings.append(x)
        encoding_im_sizes.append(np.asarray(x.get_shape().as_list()[1:-1]))

        # only downsample if we haven't reached the max
        if i >= max_time_downsample:
            ds = (2, 2, 1)
        else:
            ds = (2, 2, 2)

        if i < len(nf_enc) - 1:
            if use_maxpool:
                x = MaxPooling3D(pool_size=ds, padding='same', name='{}_enc_maxpool_{}'.format(layer_prefix, i))(x)
                #x, pool_idxs = Lambda(lambda x:._max_pool_3d_with_argmax(x, ksize=ks, strides=(2, 2, 2), padding='same'), name='{}_enc_maxpool3dwithargmax_{}'.format(layer_prefix, i))(x)
            else:
                x = Conv3D(nf_enc[i], kernel_size=ks, strides=ds,  padding='same', name='{}_enc_conv3D_{}'.format(layer_prefix, i))(x)

    if nf_dec is None:
        nf_dec = list(reversed(nf_enc[1:]))

    decoder_outputs = []
    x_encoded = x
    print(encoding_im_sizes)
    print(nf_dec)
    for ti in range(n_tasks):
        decoding_im_sizes = []
        x = x_encoded
        for i in range(len(nf_dec)):
            curr_shape = x.get_shape().as_list()[1:-1]

            print('Current shape {}, img shape {}'.format(x.get_shape().as_list(), img_shape))
            # only do upsample if we are not yet at max resolution
            if np.any(curr_shape < list(img_shape[:len(curr_shape)])):
                # TODO: fix this for time
                '''
                if i < len(nf_dec) - max_time_downsample + 1 \
                         or curr_shape[-1] >= encoding_im_sizes[-i-2][-1]:  # if we are already at the correct time scale
                    us = (2, 2, 1)
                else:
                '''
                us = (2, 2, 2)
                #decoding_im_sizes.append(encoding_im_sizes[-i-1] * np.asarray(us))

                x = UpSampling3D(size=us, name='{}_dec{}_upsamp_{}'.format(layer_prefix, ti, i))(x)

            # just concatenate the final layer here
            if i <= len(encodings) - 2:
                x = _pad_or_crop_to_shape_3D(x, np.asarray(x.get_shape().as_list()[1:-1]), encoding_im_sizes[-i-2])
                x = Concatenate(axis=-1)([x, encodings[-i-2]])
                #x = LeakyReLU(0.2)(x)
            residual_input = x

            for j in range(n_convs_per_stage):
                x = Conv3D(nf_dec[i],
                           kernel_regularizer=reg,
                           kernel_size=ks, strides=(1, 1, 1), padding='same',
                           name='{}_dec{}_conv3D_{}_{}'.format(layer_prefix, ti, i, j))(x)
                if use_dropout and i < 2:
                    x = Dropout(0.2)(x)
                if j == 0 and include_residual:
                    residual_input = x
                elif j == n_convs_per_stage - 1 and include_residual:
                    x = Add()([residual_input, x])
                x = LeakyReLU(0.2)(x)


        if do_last_conv:
            y = Conv3D(out_im_chans, kernel_size=1, padding='same', kernel_regularizer=reg,
                       name='{}_dec{}_conv3D_final'.format(layer_prefix, ti))(x)  # add your own activation after this model
        else:
            y = x
        decoder_outputs.append(y)
    # add your own activation after this model
    if n_tasks == 1:
        return y
    else:
        return decoder_outputs
