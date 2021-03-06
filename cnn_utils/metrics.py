import os
import sys

sys.path.append('../evolving_wilds')
from cnn_utils import image_utils

import keras.backend as K
import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d

class NCC():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-2, n_chans=3):
        # IMPORTANT: a higher eps (1e-2) makes training more stable
        self.win = win
        self.eps = eps
        self.n_chans = n_chans

    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        I_chans = tf.split(I, self.n_chans, axis=-1)
        J_chans = tf.split(J, self.n_chans, axis=-1)

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        # compute cross correlation
        win_size = float(np.prod(self.win))

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        total_cc = 0.
        for c in range(self.n_chans):
            I_chan = I_chans[c]
            J_chan = J_chans[c]

            # compute CC squares
            I2 = I_chan * I_chan
            J2 = J_chan * J_chan
            IJ = I_chan * J_chan

            # compute filters
            sum_filt = tf.ones([*self.win, 1, 1])
            strides = [1] * (ndims + 2)
            padding = 'VALID'

            # compute local sums via convolution
            I_sum = conv_fn(I_chan, sum_filt, strides, padding)
            J_sum = conv_fn(J_chan, sum_filt, strides, padding)
            I2_sum = conv_fn(I2, sum_filt, strides, padding)
            J2_sum = conv_fn(J2, sum_filt, strides, padding)
            IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

            u_I = I_sum / win_size
            u_J = J_sum / win_size

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            cc = cross * cross / (I_var * J_var + self.eps)
            total_cc += tf.reduce_mean(cc)
        # return negative cc.
        return total_cc / float(self.n_chans)

    def loss(self, I, J):
        return -self.ncc(I, J)

class AreaDistributionLoss(object):
    def __init__(self, ratio_mu, ratio_sigma, delta_thresh, n_frame_chans=3):
        self.ratio_mu = ratio_mu
        self.ratio_sigma = ratio_sigma
        self.delta_thresh = delta_thresh
        self.n_frame_chans = n_frame_chans


    def compute_loss(self, y_true, y_pred):
        eps = 1e-4
        # assumes that the inputs have been stacked in this order
        pred_attns, pred_frame_deltas = tf.split(y_pred, [1, self.n_frame_chans], axis=-1)

        # absolute delta
        pred_frame_deltas = tf.abs(pred_frame_deltas)

        frame_deltas_maxchans = tf.reduce_max(pred_frame_deltas, axis=-1, keepdims=True)

        frame_deltas_binarized = tf.sigmoid(100. * (frame_deltas_maxchans - self.delta_thresh))
        #frame_deltas_binarized = tf.where(frame_deltas_maxchans > self.delta_thresh, tf.ones_like(frame_deltas_maxchans), tf.zeros_like(frame_deltas_maxchans))
        
        # take only frame deltas within the attention map
        frame_deltas_binarized = frame_deltas_binarized * pred_attns

        frame_deltas_area = tf.reduce_sum(frame_deltas_binarized, axis=[1, 2, 3], keepdims=False)        
        attns_area = tf.reduce_sum(pred_attns, axis=[1, 2, 3], keepdims=False)
        delta_attn_area_ratios = frame_deltas_area / (attns_area + eps)

        ratio_log_likelihoods = -tf.square(delta_attn_area_ratios - self.ratio_mu) / (2 * self.ratio_sigma ** 2)

        return -tf.reduce_mean(ratio_log_likelihoods)


def norm_vgg(x):
    import tensorflow as tf
    eps = 1e-10
    x_norm = tf.sqrt(tf.reduce_sum(x * x, axis=-1, keep_dims=True))
    x_norm = tf.divide(x, x_norm + eps)
    return x_norm

# used for mode 'caffe'
keras_imagenet_mean = [103.939, 116.779, 123.68]

keras_imagenet_mean_torch = [0.485, 0.456, 0.406]
keras_imagenet_std_torch = [0.229, 0.224, 0.225]
# truncated vgg implementation from guha
def vgg_preprocess(arg):
    import tensorflow as tf
    z = 255.0 * tf.clip_by_value(arg, 0., 1.)
    b = z[:, :, :, 0] - 103.939
    g = z[:, :, :, 1] - 116.779
    r = z[:, :, :, 2] - 123.68
    return tf.stack([b, g, r], axis=3)

def vgg_preprocess_norm(arg):
    import tensorflow as tf
    #z = tf.clip_by_value(arg, -1., 1.)
    z = 255.0 * (arg * 0.5 + 0.5)
    b = z[:, :, :, 0] - 103.939
    g = z[:, :, :, 1] - 116.779
    r = z[:, :, :, 2] - 123.68
    return tf.stack([b, g, r], axis=3)

def vgg_preprocess_torch(arg):
    import tensorflow as tf
    #z = tf.clip_by_value(arg, -1., 1.)
    z = z * 0.5 + 0.5
    b = (z[:, :, :, 0] - 0.406) / 0.225
    g = (z[:, :, :, 1] - 0.456) / 0.224
    r = (z[:, :, :, 2] - 0.485) / 0.229
    return tf.stack([r, g, b], axis=3)

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import AveragePooling2D, Conv2D, Input, MaxPooling2D
from keras.models import Model, load_model
def vgg_isola_norm(shape=(64,64,3), normalized_inputs=False, do_preprocess=True):
    img_input = Input(shape=shape)

    if do_preprocess:
        if normalized_inputs:
            #vgg_model_file = '../evolving_wilds/cnn_utils/vgg16_normtorchtanh_ims{}-{}.h5'.format(shape[0], shape[1])
            #img = Lambda(vgg_preprocess_torch,
            #    name='lambda_preproc_torchnorm-11')(img_input)
            vgg_model_file = '../evolving_wilds/cnn_utils/vgg16_normtanh_ims{}-{}.h5'.format(shape[0], shape[1])
            img = Lambda(vgg_preprocess_norm,
                name='lambda_preproc_norm-11')(img_input)
        else:
            vgg_model_file = '../evolving_wilds/cnn_utils/vgg16_01_ims{}-{}.h5'.format(shape[0], shape[1])
            img = Lambda(vgg_preprocess,
                name='lambda_preproc_clip01')(img_input)
    else:
        vgg_model_file = '../evolving_wilds/cnn_utils/vgg16_ims{}-{}.h5'.format(shape[0], shape[1])
        img = img_input

    if os.path.isfile(vgg_model_file):
        print('Loading vgg model from {}'.format(vgg_model_file))
        return load_model(vgg_model_file,
            custom_objects={
                'vgg_preprocess_norm': vgg_preprocess_norm, 
                'vgg_preprocess': vgg_preprocess})
    
    # Block 1
    x1 = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img)
    x2 = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x1) # relu is layer 4 in torch implementation
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x2)

    # Block 2
    x4 = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x3)
    x5 = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x4) # relu is layer 9
    x6 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x5)

    # Block 3
    x7 = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x6)
    x8 = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x7)
    x9 = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x8) # relu is layer 16
    x10 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x9)

    # Block 4
    x11 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x10)
    x12 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x11)
    x13 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x12) # relu is layer 23 in torch
    x14 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x13)

    # Block 5
    x15 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x14)
    x16 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x15)
    x17 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x16) # relu is layer 30

    # isola implementation uses layer outputs 4, 9, 16, 23, 30, but need to count activations
    model = Model(inputs=[img_input], outputs=[x2, x5, x9, x13, x17])
    model_orig = VGG16(weights='imagenet', input_shape=shape, include_top=False)

    # ignore the lambda we put in for preprocessing
    vgg_layers = [l for l in model.layers if not isinstance(l, Lambda)]
    for li, l in enumerate(vgg_layers):
        weights = model_orig.layers[li].get_weights()
        l.set_weights(weights)
        print('Copying imagenet weights for layer {}: {}'.format(li, l.name))
        l.trainable = False

    if not os.path.isfile(vgg_model_file):
        model.save(vgg_model_file)

    return model



def vgg_norm(shape=(64,64,3), normalized_inputs=False):
    img_input = Input(shape=shape)

    if normalized_inputs:
        vgg_model_file = '../evolving_wilds/cnn_utils/vgg_normtanh_ims{}-{}.h5'.format(shape[0], shape[1])
        img = Lambda(vgg_preprocess_norm,
            name='lambda_preproc_norm-11')(img_input)
    else:
        vgg_model_file = '../evolving_wilds/cnn_utils/vgg_01_ims{}-{}.h5'.format(shape[0], shape[1])
        img = Lambda(vgg_preprocess,
            name='lambda_preproc_clip01')(img_input)

    if os.path.isfile(vgg_model_file):
        print('Loading vgg model from {}'.format(vgg_model_file))
        return load_model(vgg_model_file,
            custom_objects={
                'vgg_preprocess_norm': vgg_preprocess_norm, 
                'vgg_preprocess': vgg_preprocess})
    

    #img = Lambda(lambda arg:vgg_preprocess(arg, normalized=normalized_inputs),
    #    name='lambda_preproc_norm{}'.format(normalized_inputs))(img_input)

    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x1)
    x3 = AveragePooling2D((2, 2), name='block1_pool')(x2)

    x4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x3)
    x5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x4)
    x6 = AveragePooling2D((2, 2), name='block2_pool')(x5)

    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x6)
    x8 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x7)
    x9 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x8)
    x10 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x9)
    x11 = AveragePooling2D((2, 2), name='block3_pool')(x10)

    x12 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x11)
    x13 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x12)
    x14 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x13)
    x15 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x14)
    x16 = AveragePooling2D((2, 2), name='block4_pool')(x15)

    x17 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x16)
    x18 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x17)
    x19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x18)
    x20 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x19)
    x21 = AveragePooling2D((2, 2), name='block5_pool')(x20)

    model = Model(inputs=[img_input], outputs=[x2, x5, x9, x14, x19])
    model_orig = VGG19(weights='imagenet', input_shape=shape, include_top=False)

    # ignore the lambda we put in for preprocessing
    vgg_layers = [l for l in model.layers if not isinstance(l, Lambda)]
    for li, l in enumerate(vgg_layers):
        weights = model_orig.layers[li].get_weights()
        l.set_weights(weights)
        print('Copying imagenet weights for layer {}: {}'.format(li, l.name))
        l.trainable = False

    if not os.path.isfile(vgg_model_file):
        model.save(vgg_model_file)

    return model


#from networks import affine_networks
#from networks import transform_network_utils
from keras.layers import Lambda
class VggFeatLoss(object):
    def __init__(self, feat_net, dist='l2'):
        self.feat_net = feat_net
        self.dist = dist

    def compute_loss(self, y_true, y_pred):
        import tensorflow as tf
        # just preprocess as a part of the model
        n_feature_layers = len(self.feat_net.outputs)
        x1 = self.feat_net(y_true)
        x2 = self.feat_net(y_pred)

        loss = []

        for li in range(n_feature_layers):
            x1_l = x1[li]
            x2_l = x2[li]

            # unit normalize in channels dimension
            #x1_norm = tf.sqrt(tf.reduce_sum(x1_l * x1_l, axis=-1, keep_dims=True))  # b x h x w x 1
            #x2_norm = tf.sqrt(tf.reduce_sum(x2_l * x2_l, axis=-1, keep_dims=True))

            #x1_l_norm = tf.divide(x1_l, x1_norm)  # b x h x w x c
            #x2_l_norm = tf.divide(x2_l, x2_norm)
            x1_l_norm = norm_vgg(x1_l)
            x2_l_norm = norm_vgg(x2_l)

            hw = tf.shape(x1_l)[1] * tf.shape(x1_l)[2]

            if self.dist == 'l1':
                d = tf.reduce_sum(tf.abs(x1_l_norm - x2_l_norm), [1, 2, 3])  # bx1
            else:
                d = tf.reduce_sum(tf.square(x1_l_norm - x2_l_norm), [1, 2, 3])  # bx1
            d_mean = tf.divide(d, tf.cast(hw, tf.float32))

            if li == 0:
                loss = d_mean
            else:
                loss = loss + d_mean
        return loss

class MinLossOverSamples(object):
    def __init__(self, n_samples, pred_shape,
                     loss_name=None,
                     loss_fn=None,
                     ):
        self.n_samples = n_samples
        self.loss_name = loss_name
        self.loss_fn = loss_fn
        self.pred_shape = pred_shape

    def compute_loss(self, y_true, y_pred):
        y_true_samples = tf.reshape(y_true, [-1, 1] + list(self.pred_shape))#, [1, self.n_samples] + [1] * len(self.pred_shape))
        if self.loss_name is not None:
            y_pred_samples = tf.reshape(y_pred, [-1, self.n_samples] + list(self.pred_shape))
            # average over every dimension except batch
            if 'l1' in self.loss_name:
                loss_vals = tf.reduce_mean(tf.abs(y_pred_samples - y_true_samples), axis=list(range(2, len(self.pred_shape)+2)))
            else:
                loss_vals = tf.reduce_mean(tf.square(y_pred_samples - y_true_samples), axis=list(range(2, len(self.pred_shape)+2)))
        else:
            # tile y_true samples so that they are the same shape as y_pred
            y_true = tf.reshape(
                tf.tile(y_true_samples, [1, self.n_samples] + [1] * len(self.pred_shape)), [-1] + list(self.pred_shape))
            loss_vals = self.loss_fn(y_true=y_true, y_pred=y_pred)
        print('loss shape {}'.format(loss_vals.get_shape()))
        # reshape to batch x n_samples
        loss_per_sample = tf.reshape(loss_vals, [-1, self.n_samples], name='reshape_batches_samples')
        min_loss = tf.reduce_min(loss_per_sample, axis=1)
        return tf.reduce_mean(min_loss)



class SoftSuperpixelLoss(object):
    def __init__(self, img_shape, lambdas=[1., 1., 1., 1., 1.], sigma_norm=0.2):
        h, w = img_shape[:2]
        n_dims = 5  # x y rgb

        # to normalize space and color
        self.lambdas_norm = tf.constant(
            np.tile(np.reshape([1. / w, 1. / h, 1., 1., 1.], (1, 1, n_dims)), (1, h * w, 1)), dtype=tf.float32)

        xs, ys = np.meshgrid(np.linspace(0, w, w, endpoint=False).astype(int),
                                    np.linspace(0, h, h, endpoint=False).astype(int))
        xs = np.reshape(xs, (1, -1, 1))
        ys = np.reshape(ys, (1, -1, 1))

        self.xs = tf.constant(xs, dtype=tf.int32)
        self.ys = tf.constant(ys, dtype=tf.int32)
        self.lambdas = tf.constant(lambdas / np.sum(lambdas), dtype=tf.float32)
        self.sigma_norm = sigma_norm


    def compute_center_coords(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        h = tf.shape(y_pred)[1]
        w = tf.shape(y_pred)[2]
        n_chans = tf.shape(y_pred)[3]
        n_dims = 5

        # weighted center of mass
        x = tf.cast(tf.tile(tf.reshape(self.xs, [1, h, w]), [batch_size, 1, 1]), tf.float32)
        y = tf.cast(tf.tile(tf.reshape(self.ys, [1, h, w]), [batch_size, 1, 1]), tf.float32)

        eps = 1e-8
        # grayscale
        pred_gray = tf.reduce_mean(y_pred, axis=-1)  # should be batch_size x h x w
        # normalize
        pred_gray = pred_gray - tf.reduce_min(pred_gray, axis=[1, 2], keepdims=True)
        pred_gray = pred_gray / (eps + tf.reduce_max(pred_gray, axis=[1, 2], keepdims=True))
        pred_gray = tf.clip_by_value(pred_gray, 0., 1.)

        # make each of these (batch_size, 1)
        weighted_x = tf.round(tf.expand_dims(
            tf.reduce_sum(x * pred_gray, axis=[1, 2]) / (eps + tf.reduce_sum(pred_gray, axis=[1, 2])), axis=-1))
        weighted_y = tf.round(tf.expand_dims(
            tf.reduce_sum(y * pred_gray, axis=[1, 2]) / (eps + tf.reduce_sum(pred_gray, axis=[1, 2])), axis=-1))
        batch_indices = tf.reshape(tf.linspace(0., tf.cast(batch_size, tf.float32) - 1., batch_size), [batch_size, 1])
        indices = tf.cast(tf.concat([batch_indices, weighted_y, weighted_x], axis=-1), tf.int32)
        #center_rgb = transform_network_utils.interpolate([y_true,  weighted_x, weighted_y], constant_vals=1.)
        center_rgb = tf.gather_nd(y_true, indices)
        center_rgb = tf.reshape(center_rgb, [batch_size, n_chans])

        center_point_xyrgb = tf.concat([
                        weighted_x, weighted_y, center_rgb
                    ], axis=-1)

        return pred_gray, center_point_xyrgb


    def compute_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        h = tf.shape(y_true)[1]
        w = tf.shape(y_true)[2]
        n_chans = tf.shape(y_true)[3]
        n_pixels = h * w
        n_dims = 5
        eps = 1e-8

        # indices in batch, row, column format
        #y_pred_norm, center_x, center_y = self.compute_center_coords(y_pred)
        y_pred_norm, center_point_xyrgb = self.compute_center_coords(y_true, y_pred)
        center_point_xyrgb = tf.tile(
            tf.reshape(center_point_xyrgb, [batch_size, 1, n_dims]), (1, h * w, 1))
        #center_x = tf.reshape(center_x, [batch_size])
        #center_y = tf.reshape(center_y, [batch_size])
        # make a batch_size x 3 matrix so we can index into the batch, r, c dimensions
        #center_rgb = tf.gather_nd(y_true, center_point_bxy)  # should be batch_size x 3
        true_rgbs = tf.reshape(y_true, [batch_size, n_pixels, n_chans])

        im_coords = tf.concat([
            tf.cast(tf.tile(self.xs, [batch_size, 1, 1]), tf.float32),
            tf.cast(tf.tile(self.ys, [batch_size, 1, 1]), tf.float32),
            true_rgbs
        ], axis=-1)

        # compute normalized distance, and weight using lambdas
        pixel_dists = ((im_coords - center_point_xyrgb) * self.lambdas_norm) ** 2 * self.lambdas
        soft_pixel_affinities = (1. - tf.exp(tf.reduce_sum(-0.5 * pixel_dists / self.sigma_norm ** 2, axis=-1)))
        soft_pixel_affinities = tf.reshape(soft_pixel_affinities, [batch_size, h, w])  # weight mask

        return soft_pixel_affinities * y_pred_norm


class WeightedCategoricalCrossEntropy(object):
    def __init__(self, class_weights):
        self.class_weights = tf.reshape(
            tf.constant(class_weights, dtype=tf.float32), [1, 1, 1, len(class_weights)])
        self.n_classes = len(class_weights)

    def compute_loss(self, y_true, y_pred):
        #weights = K.tile(self.class_weights, [tf.shape(y_true)[0], tf.shape(y_true)[1], tf.shape(y_true)[2], 1])
        # softmax
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calc
        loss = y_true * K.log(y_pred) * self.class_weights
        loss = -K.sum(loss, -1)
        return loss
        

class TransformedLoss(object):
    '''
    Computes a loss function against a warped ground truth
    '''
    def __init__(self,
                     transform_layer_output,
                     loss_fn,
                     pad_val=1.):
        self.transform_layer_output = transform_layer_output
        self.loss_fn = loss_fn
        self.pad_val = pad_val

    def compute_loss(self, y_true, y_pred):
        # flatten time and channels dimension together
        y_true = tf.reshape(y_true, 
            [tf.shape(y_true)[0], tf.shape(y_true)[1], tf.shape(y_true)[2], -1])
        y_pred = tf.reshape(y_pred,
            [tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2], -1])

        # warp the ground truth
        y_true_warped = transform_network_utils.affineWarp(
            y_true, self.transform_layer_output, self.pad_val)
        return self.loss_fn(y_true_warped, y_pred)


class PatchLoss(object):
    def __init__(self, n_patches, img_shape, patch_size, mask_output=None, agg_type='median', norm_type='l1'):
        self.n_patches = n_patches
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.agg_type = agg_type
        self.norm_type = norm_type
        self.mask_output = mask_output

    def compute_loss(self, y_true, y_pred):
        patches_true = affine_networks._get_patches(y_true, self.img_shape, self.patch_size)
        patches_pred = affine_networks._get_patches(y_pred, self.img_shape, self.patch_size)

        if self.mask_output is not None:
            patches_mask = affine_networks._get_patches(self.mask_output, self.img_shape, self.patch_size)

            if self.norm_type == 'l1':
                losses = tf.reduce_sum(tf.abs(patches_true - patches_pred), axis=[1, 2, 3])
            elif self.norm_type == 'l2':
                losses = 0.5 * tf.reduce_sum(tf.square(patches_true - patches_pred), axis=[1, 2, 3])

            losses = losses / (1e-8 + tf.reduce_sum(patches_mask, axis=[1, 2, 3]))
            patch_maxes = tf.reduce_max(patches_mask, axis=[1, 2, 3])
        else:    
            if self.norm_type == 'l1':
                losses = tf.reduce_mean(tf.abs(patches_true - patches_pred), axis=[1, 2, 3])  # take the mean over everything but patches
            elif self.norm_type == 'l2':
                losses = 0.5 * tf.reduce_mean(tf.square(patches_true - patches_pred), axis=[1, 2, 3])
            patch_maxes = tf.reduce_max(patches_true, axis=[1, 2, 3])  # make batch_size x n_patches

        if self.agg_type == 'median':
            med_num = tf.shape(losses)[-1] / 2
            #med_num = len(losses) / 2
            vals, idxs = tf.nn.top_k(losses, self.n_patches/2)
            return vals[0]
        elif self.agg_type == 'median-nonzero':
            keep_patches = tf.greater(patch_maxes, 1e-2)
            keep_losses = tf.boolean_mask(losses, keep_patches)
            n_losses = tf.shape(keep_losses)[-1]
            med_num = tf.shape(keep_losses)[-1] / 2
            #med_num = len(losses) / 2
            vals, idxs = tf.nn.top_k(keep_losses, n_losses)  # sort losses
            return tf.reduce_mean(vals[-med_num:])  # take the bottom half (lowest) losses
        else:
            print('Computing mean L1 over all patches')
            return tf.reduce_mean(losses)

class SummedLosses(object):
    def __init__(self, loss_fns, loss_weights):
        assert len(loss_fns) == len(loss_weights)
        self.loss_fns = loss_fns
        self.loss_weights = loss_weights

    def compute_loss(self, y_true, y_pred):
        total_loss = 0
        for li in range(len(self.loss_fns)):
            total_loss += self.loss_weights[li] * tf.reduce_mean(self.loss_fns[li](y_true, y_pred))
        return total_loss


class NegGammaProbLoss(object):
    def __init__(self, gamma_a, gamma_b):
        self.gd = tf.distributions.Gamma(
            tf.constant(gamma_a, dtype=tf.float32), 
            tf.constant(gamma_b, dtype=tf.float32)
        )
    
    def compute_loss(self, _, y_pred):
        total_prob = 0

        # compute l1 norm of each frame, and then take the mean over all pixels, frames and color channels
        # don't take the delta of the first frame since we actually want it to be 0
        y_pred = tf.reduce_mean(tf.abs(y_pred[:, :, :, 1:, :]), [1, 2, 3, 4])
        total_prob = self.gd.prob(y_pred)
        return -tf.reduce_mean(total_prob)


class NegHistogramLoss(object):
    def __init__(self, hist, bin_edges):
        self.ref_hist = hist
        self.ref_bin_edges = tf.constant(bin_edges, dtype=tf.float32)
        self.n_bins = hist.shape[0]

    def compute_loss(self, _, y_pred):
        total_prob = 0
        # don't take the delta of the first frame since we actually want it to be 0
        y_pred = tf.reduce_mean(tf.abs(y_pred[:, :, :, 1:, :]), [1, 2, 3, 4])
        #return tf.reduce_sum(y_pred)
        for i in range(self.n_bins):
            lt_binmax = tf.less(y_pred, self.ref_bin_edges[i+1])
            gt_binmin = tf.greater_equal(y_pred, self.ref_bin_edges[i])
            in_bin = tf.cast(lt_binmax, tf.float32) * tf.cast(gt_binmin, tf.float32)
            total_prob += tf.reduce_sum(in_bin) * self.ref_hist[i]
        return -total_prob            


class TimeSliceLoss(object):
    def __init__(self, time_axis, loss_fn, 
            slice_idxs=[],
            n_frames=30,
            compute_mean = True):
        self.time_axis = time_axis
        self.loss_fn = loss_fn
        self.compute_mean = compute_mean    
        self.n_frames = n_frames
        self.slice_len = len(slice_idxs)
        self.slice_idxs = tf.constant(slice_idxs, dtype=tf.int32)
        print(slice_idxs)
        print(self.slice_idxs)

    def compute_loss(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [
            tf.shape(y_true)[0], tf.shape(y_true)[1], tf.shape(y_true)[2], self.n_frames, 3])
        y_pred = tf.reshape(y_pred, [
            tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2], self.n_frames, 3])

        y_true = tf.gather(y_true, indices=self.slice_idxs, axis=self.time_axis)
        y_pred = tf.gather(y_pred, indices=self.slice_idxs, axis=self.time_axis)
        total_loss = self.loss_fn(y_true, y_pred)
        
        '''
        true_frames = tf.unstack(y_true, num=self.n_frames, axis=self.time_axis)
        pred_frames = tf.unstack(y_pred, num=self.n_frames, axis=self.time_axis)

        total_loss = 0
        for t in self.slice_idxs:
            loss = self.loss_fn(true_frames[t], pred_frames[t])
            total_loss += loss
        '''
        if self.compute_mean:
            total_loss /= float(self.slice_len)
        return total_loss


class TimeSummedLoss(object):
    def __init__(self, loss_fn, weights_output=None,
                     time_axis=-2, compute_mean=True, pad_amt=None,
                    do_reshape_to=None,
                     compute_over_frame_idxs=None,
                     ):
        self.time_axis = time_axis
        self.loss_fn = loss_fn
        self.weights_output = weights_output # in case we want to predict the weight for each time step
        self.compute_mean = compute_mean    
        self.pad_amt = pad_amt
        self.include_frames = compute_over_frame_idxs
        self.do_reshape_to = do_reshape_to

    def compute_loss(self, y_true, y_pred):
        if self.do_reshape_to is not None: # reshape a flattened video
            y_true = tf.reshape(y_true, [-1] + list(self.do_reshape_to))
            y_pred = tf.reshape(y_pred, [-1] + list(self.do_reshape_to))

        if self.pad_amt is not None:
            y_true = tf.pad(y_true, paddings=self.pad_amt, constant_values=1.)
            y_pred = tf.pad(y_pred, paddings=self.pad_amt, constant_values=1.)

        n_frames = y_pred.get_shape().as_list()[self.time_axis]
        if self.include_frames is None:
            include_frames = list(range(n_frames))
        else:
            include_frames = self.include_frames

        # TODO: switch to tf.map_fn?
        true_frames = tf.unstack(y_true, num=n_frames, axis=self.time_axis)
        pred_frames = tf.unstack(y_pred, num=n_frames, axis=self.time_axis)

        if self.weights_output is not None:
            weights_list = tf.unstack(self.weights_output, num=n_frames, axis=-1)
        else:
            weights_list = None

        total_loss = 0
        for t in include_frames:
            loss = self.loss_fn(y_true=true_frames[t], y_pred=pred_frames[t])
            if weights_list is not None:
                curr_weights = tf.expand_dims(weights_list[t], axis=-1)
                loss = tf.multiply(loss, curr_weights)

            total_loss += loss

        if self.compute_mean:
            total_loss /= float(n_frames)

        return total_loss



def top_k_cat_acc(k=5):
    def compute_acc(y_true,y_pred):
         return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)
    return compute_acc


class VoxelmorphMetrics_bound(object):
    def __init__(self, flow_mean, flow_logvar, alpha=1.):
        self.alpha = alpha
        self.flow_mean = flow_mean
        self.flow_logvar = flow_logvar
        self.vm_metrics = VoxelmorphMetrics(self.alpha)

    def kl_loss(self, _, y_pred):
        mean = self.flow_mean
        log_sigma = self.flow_logvar
        
        # compute the degree matrix. If this has already happened
        # should only compute this once!
        # also need to check that this works!
        # z = K.ones((1, ) + vol_size + (3, ))
        sz = log_sigma.get_shape().as_list()[1:] 
        z = K.ones([1] + sz)

        filt = np.zeros((3,3,2,2))
        for i in range(2):
            filt[[0,2],1,i,i] = 1
            filt[1,[0,2],i,i] = 1
        filt_tf = tf.convert_to_tensor(filt,dtype=tf.float32)
        D = tf.nn.conv2d(z, filt_tf, [1,1,1,1],"SAME")
        D = K.expand_dims(D, 0)

        sigma_terms = (self.alpha * D * tf.exp(log_sigma) - log_sigma)

        prec_terms = self.alpha * self.vm_metrics.kl_prec_term_manual(_, mean) # note needs 0.5 twice, one here, one below
        kl = 0.5 * tf.reduce_mean(sigma_terms,[1,2]) + 0.5 * prec_terms
        return kl


class VoxelmorphMetrics(object):
    def __init__(self, alpha=1.):
        self.alpha = alpha
        return None

    def kl_loss(self, _, y_pred):
        mean = y_pred[:,:,:,0:2]
        log_sigma = y_pred[:,:,:,2:]
        
        # compute the degree matrix. If this has already happened
        # should only compute this once!
        # also need to check that this works!
        # z = K.ones((1, ) + vol_size + (3, ))
        sz = log_sigma.get_shape().as_list()[1:] 
        z = K.ones([1] + sz)

        filt = np.zeros((3,3,2,2))
        for i in range(2):
            filt[[0,2],1,i,i] = 1
            filt[1,[0,2],i,i] = 1
        filt_tf = tf.convert_to_tensor(filt,dtype=tf.float32)
        D = tf.nn.conv2d(z, filt_tf, [1,1,1,1],"SAME")
        D = K.expand_dims(D, 0)

        sigma_terms = (self.alpha * D * tf.exp(log_sigma) - log_sigma)

        prec_terms = 0.5 * self.alpha * self.kl_prec_term_manual(_, mean) # note needs 0.5 twice, one here, one below
        kl = 0.5 * tf.reduce_mean(sigma_terms,[1,2]) + 0.5 * prec_terms
        return kl

    def smoothness_precision_loss(self, y_true, y_pred):
        # assumes that sigma is very small, so we don't need to deal with it
#        sz = log_sigma.get_shape().as_list()[1:] 
#        z = K.ones([1] + sz)

#        filt = np.zeros((3,3,2,2))
#        for i in range(2):
#            filt[[0,2],1,i,i] = 1
#            filt[1,[0,2],i,i] = 1
#        filt_tf = tf.convert_to_tensor(filt,dtype=tf.float32)
#        D = tf.nn.conv2d(z, filt_tf, [1,1,1,1],"SAME")
#        D = K.expand_dims(D, 0)

        prec_terms = 0.5 * self.alpha * self.prec_term_manual(y_true, y_pred) # note needs 0.5 twice, one here, one below
#        kl = 0.5 * prec_terms
        return prec_terms
        
    def smoothness_precision_loss_zeromean(self, _, y_pred):
        return 0.5 * self.alpha * self.kl_prec_term_manual(_, y_pred)

    def kl_loss_log_sigma(self,_,y_pred):
        log_sigma = y_pred
        kl = 0.5 * tf.reduce_mean(tf.exp(log_sigma) - log_sigma,[1,2])
        return kl

    # this version does not assume zero-mean
    def prec_term_manual(self,y_true,y_pred):
        """ 
        this is the more manual implementation of the precision matrix term
            P = D - A
            mu * P * mu
        where D is the degree matrix and A is the adjacency matrix
            mu * P * mu = sum_i mu_i sum_j (mu_i - mu_j)
        where j are neighbors of i
        """
        # i is at (x,y), j is at (x-1, y)
        dy = (y_pred[:,1:,:,:] - y_true[:,1:,:,:]) * (y_pred[:,1:,:,:] - y_true[:,1:,:,:] - y_pred[:,:-1,:,:] + y_true[:,:-1,:,:])
        # i is at (x,y), j is at (x, y-1)
        dx = (y_pred[:,:,1:,:] - y_true[:,:,1:,:])  * (y_pred[:,:,1:,:] - y_true[:,:,1:,:] - y_pred[:,:,:-1,:] + y_true[:,:,:-1,:])

        # i is at (x,y), j is at (x+1, y)
        dy2 = (y_pred[:,:-1,:,:] - y_true[:,:-1,:,:]) * (y_pred[:,:-1,:,:] - y_true[:,:-1,:,:] - y_pred[:,1:,:,:] + y_true[:,1:,:,:])

        # i is at (x,y), j is at (x, y+1)
        dx2 = (y_pred[:,:,:-1,:] - y_true[:,:,:-1,:]) * (y_pred[:,:,:-1,:] - y_true[:,:,:-1,:] - y_pred[:,:,1:,:] + y_true[:,:,1:,:])

        d = tf.reduce_mean(dx) + tf.reduce_mean(dy) + tf.reduce_mean(dy2) + tf.reduce_mean(dx2) 
        return d

    def kl_prec_term_manual(self,y_true,y_pred):
        """ 
        this is the more manual implementation of the precision matrix term
            P = D - A
            mu * P * mu
        where D is the degree matrix and A is the adjacency matrix
            mu * P * mu = sum_i mu_i sum_j (mu_i - mu_j)
        where j are neighbors of i
        """
        dy = y_pred[:,1:,:,:] * (y_pred[:,1:,:,:] - y_pred[:,:-1,:,:])
        dx = y_pred[:,:,1:,:] * (y_pred[:,:,1:,:] - y_pred[:,:,:-1,:])
        dy2 = y_pred[:,:-1,:,:] * (y_pred[:,:-1,:,:] - y_pred[:,1:,:,:])
        dx2 = y_pred[:,:,:-1,:] * (y_pred[:,:,:-1,:] - y_pred[:,:,1:,:])

        d = tf.reduce_mean(dx) + tf.reduce_mean(dy) + tf.reduce_mean(dy2) + tf.reduce_mean(dx2) 
        return d

    def localNorm(I):
        I2 = tf.square(I)+K.epsilon()
        filt = tf.ones([9,9,1,1])
        I_sum = tf.nn.conv3d(I,filt,[1,1,1,1], "SAME")
        I2_sum = tf.nn.conv3d(I2,filt,[1,1,1,1], "SAME")

        win_size = 9*9.0
        u_I = I_sum/win_size

        sigma = I2_sum-u_I*u_I*win_size
        sigma = K.sqrt(sigma/(win_size-1)) + 0.01

        I_norm = (I-u_I)/sigma

        return I_norm

class LearnedSigmaL2(object):
    def __init__(self,
                     var_map=None,
                     logvar_map=None):
        self.var_map = var_map
        self.logvar_map = logvar_map


    def compute_sigma_reg(self, y_true, y_pred):
        if self.logvar_map is not None:
            logvar_map = self.logvar_map
        elif self.var_map is not None:
            logvar_map = K.log(self.var_map + 1e-8)

        # we will scale later to K.sum
        return  0.5 * K.clip(logvar_map, -100, 100) 


    def compute_l2_loss(self, y_true, y_pred):
        reg = 1e-2
        if self.logvar_map is not None:
            var_map = K.exp(self.logvar_map) + 1e-8
        elif self.var_map is not None:
            var_map = self.var_map + 1e-8

        # we will scale later to K.sum
        return  0.5 * tf.divide(K.square(y_pred - y_true) + reg, var_map)



class WeightedLengthDistributionLoss(object):
    def __init__(self, length_mu, length_sigma, is_done_model):
        self.length_mu = length_mu
        self.length_sigma = length_sigma
        self.is_done_model = is_done_model

    def compute_loss(self, y_true, y_pred):
        eps = 1e-4
        # assumes that the inputs have been stacked in this order
        #is_done_preds = self.is_done_model(y_pred)

        seq_len = y_pred.get_shape().as_list()[-1]
        seq_lens = tf.reshape(tf.range(0, seq_len, dtype=tf.float32), [1, 1, seq_len])
        seq_len_log_likelihoods = -tf.square(seq_lens - self.length_mu) / (2 * self.length_sigma ** 2)
        weighted_ll = tf.log(y_pred + eps) + seq_len_log_likelihoods

        # weight log likelihoods by probability that the sequence is done
        return -tf.reduce_mean(weighted_ll)


class VAE_metrics_categorical(object):
    """
    Losses for variational auto-encoders
    """

    def __init__(self,
                 n_categories,
                 prior=None,
                 axis=1):
        self.n_categories = n_categories
        self.axis = axis
        self.prior = prior

    def kl_categorical(self, y_true, y_pred):
        eps = 1e-8
        if self.prior is None:
            return K.sum(y_pred * (K.log(y_pred + eps) - tf.log(1. / self.n_categories)), axis=self.axis)
        else:
            return K.sum(y_pred * (K.log(y_pred + eps) - K.log(self.prior + eps)), axis=self.axis)

class VAE_metrics(object):
    """
    Losses for variational auto-encoders
    """

    def __init__(self,
                     var_target=None,
                 logvar_target=None,
                     mu_target=None,
                     axis=1):
#        self.log_var_pred = log_var_pred
#        self.mu_pred = mu_pred


        self.var_target = var_target
        self.logvar_target = logvar_target
                                
        self.mu_target = mu_target

        self.axis = axis


    def kl_log_sigma(self, y_true, y_pred):
        """
        kl_log_sigma terms of the KL divergence
        """
        eps = 1e-8

        logvar_pred = y_pred
        var_pred = K.exp(y_pred)

        if self.var_target is None and self.logvar_target is not None:
            var_target = K.exp(self.logvar_target)
        elif self.var_target is not None:
            var_target = self.var_target
        elif self.var_target is None and self.logvar_target is None:
            var_target = y_true

        kl_sigma_out = 0.5 * K.sum(
            (var_pred / (var_target + eps)) \
            + K.log(var_target)
            - logvar_pred \
            - 1, axis=self.axis)
        return kl_sigma_out


    def kl_mu(self, y_true, y_pred):
        """
        kl_mu terms of the KL divergence
        y_pred should be mu_out
        """
        eps = 1e-8

        if self.var_target is None and self.logvar_target is not None:
            var_target = K.exp(self.logvar_target)
        elif self.var_target is not None:
            var_target = self.var_target
        elif self.var_target is None and self.logvar_target is None:
            var_target = y_true

        # TODO: we cant have both self.mu_target is None and slef.var_target is None
        if self.mu_target is None:
            mu_target = y_true
        else:
            mu_target = self.mu_target

        kl_mu_out = 0.5 * K.sum(
                K.square(y_pred - mu_target) / (var_target + eps),
            axis=self.axis)
        return kl_mu_out

    # def complete_loss(self, y_true, y_pred):
    #     """
    #     Using implementation from here as initial guidance:
    #     https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
    #     To balance the terms, there is a natural term that should be computed,
    #     which is not covered below
    #     Comments from blog post:
    #     Calculate loss = reconstruction loss + KL loss for each data in minibatch
    #     """
    #     # E[log P(X|z)]
    #     recon = K.sum(keras.losses.mean_squared_error(y_true, y_pred))
    #
    #     # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    #     term_kl = K.exp(self.log_sigma_out) + K.square(self.mu_out) - 1. - self.log_sigma_out
    #     kl = 0.5 * K.sum(term_kl, axis=self.axis)
    #
    #     return recon + kl

class BoundMaskedLoss(object):
    def __init__(self, mask_output, loss_fn,
            invert_mask=False, compute_mean=True):
        self.mask_output = mask_output
        if invert_mask:
            self.mask_output = 1 - mask_output
        self.compute_mean = compute_mean
        self.loss_fn = loss_fn

    def compute_loss(self, y_true, y_pred):
        loss = self.loss_fn(self.mask_output * y_true, self.mask_output * y_pred)

        if self.compute_mean:
            return loss * tf.cast(tf.size(y_true), tf.float32) / (1e-8 + K.sum(self.mask_output))
        else:
            return loss

class BoundMaskedL2(object):
    def __init__(self, mask_output, 
            invert_mask=False, compute_mean=True, norm_type='l2'):
        self.mask_output = mask_output
        if invert_mask:
            self.mask_output = 1 - mask_output
        self.compute_mean = compute_mean
        self.norm_type = norm_type

    def compute_loss(self, y_true, y_pred):
        if self.norm_type == 'l2':
            loss = K.sum(self.mask_output * (K.square(y_true - y_pred)))
        else:
            loss = K.sum(self.mask_output * (K.abs(y_true - y_pred)))

        if self.compute_mean:
            # assumes that the loss takes the average over the input
            return loss / (1e-8 + K.sum(self.mask_output))          
        else:
            return loss


def masked_l2_loss( y_true, y_pred ):
    y_true_im, y_true_mask = tf.split( y_true, [3,1], axis=3)
    return K.sum( K.square( y_pred - y_true_im ) * y_true_mask ) / K.sum( y_true_mask )

def kl_loss( y_true, y_pred ):
    z_mean,z_log_var = tf.split( y_pred, 2, axis=-1)
    return -0.5 * K.sum( 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1, keepdims = True)


def gradient_loss_l2(n_dims):

    def compute_loss(y_true,y_pred):
        loss = 0.
        for d in range(n_dims):
            # we use x to indicate the current spatial dimension, not just the first
            dydx = tf.abs(tf.gather(y_pred, tf.range(1, tf.shape(y_pred)[d + 1]), axis=d + 1) \
                    - tf.gather(y_pred, tf.range(0, tf.shape(y_pred)[d + 1] - 1), axis=d + 1))

            # average across spatial dims and color channels
            loss += tf.reduce_mean(dydx * dydx)
        return loss / float(n_dims)
    return compute_loss
    '''
    dy = tf.abs(y_pred[:,1:,:,:] - y_pred[:,:-1,:,:])    
    dx = tf.abs(y_pred[:,:,1:,:] - y_pred[:,:,:-1,:])

    dy = dy * dy
    dx = dx * dx
    d = tf.reduce_mean(dx)+tf.reduce_mean(dy)
    return d/2.0
    '''

# assumes time is the 3rd dimension, but we need to separate out the channels
def gradient_temporal_loss_l2(n_chans=3, norm=2):
    def compute_loss(_, y_pred):
        #y_true = tf.reshape(y_true, [tf.shape(y_true)[0], tf.shape(y_true)[1], tf.shape(y_true)[2], -1, n_chans)
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2], -1, n_chans])

        dt = tf.abs(y_pred[:,:,:,1:] - y_pred[:,:,:,:-1])    
        if norm == 2:
            dt2 = dt * dt
        else:
            dt2 = dt
        d = tf.reduce_mean(dt2)
        return d
    return compute_loss


class laplacian_reg(object):
    def __init__(self, n_chans):
        self.n_chans = n_chans
        laplacian_filter = tf.constant([[0, -1, 0],
                                                  [-1, 4, -1],
                                                  [0, -1, 0],
                                                  ], dtype=tf.float32)
        self.laplacian_filter = tf.expand_dims(tf.expand_dims(laplacian_filter, -1), -1)

    def compute_laplacian_l2(self, y_true, y_pred):
        # roll any dims after 3 into the channels dimension
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2], -1])
        pred_perchan = tf.unstack(y_pred, num=self.n_chans, axis=-1)
        loss = 0.
        for c in range(self.n_chans):
            u = tf.expand_dims(pred_perchan[c], -1)

            ddu = tf.nn.conv2d(u, self.laplacian_filter, [1, 1, 1, 1], "VALID")

            ddu2 = ddu * ddu
            loss += tf.reduce_mean(ddu2) / 2.
        return loss / float(self.n_chans)


class SpatialSegmentSmoothness(object):
    def __init__(self, n_chans, n_dims,
                 warped_contours_layer_output=None,
                 lambda_i=1.
                 ):
        self.n_dims = n_dims
        self.warped_contours_layer_output = warped_contours_layer_output
        self.lambda_i = lambda_i

    def compute_loss(self, y_true, y_pred):
        loss = 0
        segments_mask = 1. - self.warped_contours_layer_output

        for d in range(self.n_dims):
            # we use x to indicate the current spatial dimension, not just the first
            dCdx = tf.gather(y_pred, tf.range(1, tf.shape(y_pred)[d + 1]), axis=d + 1) \
                     - tf.gather(y_pred, tf.range(0, tf.shape(y_pred)[d + 1] - 1), axis=d + 1)

            # average across spatial dims and color channels
            loss += tf.reduce_mean(tf.abs(dCdx * tf.gather(segments_mask, tf.range(1, tf.shape(y_pred)[d+1]), axis=d+1)))
        return loss



class SpatialIntensitySmoothness(object):
    def __init__(self, n_chans, n_dims, 
            use_true_gradients=True, pred_image_output=None,
            lambda_i=1.
        ):
        self.n_dims = n_dims
        self.use_true_gradients = use_true_gradients
        self.pred_image_output = pred_image_output
        self.lambda_i = lambda_i

    def compute_loss(self, y_true, y_pred):
        loss = 0

        for d in range(self.n_dims):
            # we use x to indicate the current spatial dimension, not just the first
            dCdx = tf.gather(y_pred, tf.range(1, tf.shape(y_pred)[d + 1]), axis=d + 1) \
                     - tf.gather(y_pred, tf.range(0, tf.shape(y_pred)[d + 1] - 1), axis=d + 1)

            if self.use_true_gradients:
                dIdx = tf.abs(tf.gather(y_true, tf.range(1, tf.shape(y_true)[d + 1]), axis=d + 1) \
                        - tf.gather(y_true, tf.range(0, tf.shape(y_true)[d + 1] - 1), axis=d + 1))

            else:
                dIdx = self.lambda_i * tf.abs(tf.gather(self.pred_image_output, tf.range(1, tf.shape(y_true)[d + 1]), axis=d + 1) \
                        - tf.gather(self.pred_image_output, tf.range(0, tf.shape(y_true)[d + 1] - 1), axis=d + 1))

            # average across spatial dims and color channels
            loss += tf.reduce_mean(tf.abs(dCdx * tf.exp(-dIdx)))
        return loss

def bounded_neg_mean_loss(y_true, y_pred):
    return K.sigmoid(-K.mean(y_pred))


class DiscriminatorScore(object):
    # regular GAN loss
    def __init__(self, disc_model, loss_fn, target_score):
        self.disc_model = disc_model
        self.loss_fn = loss_fn
        self.target_score = target_score

    def compute_loss(self, y_true, y_pred):
        self.disc_model.trainable = False
        for l in self.disc_model.layers:
            l.trainable = False

        disc_score = self.disc_model(y_pred)
        #disc_target = tf.ones(disc_score.get_shape().as_list()) * self.target_score
        # TODO: might break because y_true is not the same size as y_pred
        return self.loss_fn(self.target_score, disc_score)


class CriticScore(object):
    # WGAN loss
    def __init__(self, critic_model, loss_fn=None, target_score=None):
        self.critic_model = critic_model
        self.critic_model.trainable = False
        for l in self.critic_model.layers:
            l.trainable = False

        self.loss_fn = loss_fn
        self.target_score = target_score
        
    def compute_loss(self, y_true, y_pred):
        critic_score = self.critic_model(y_pred)

        if self.loss_fn is None:
            return -tf.reduce_mean(critic_score)
        elif self.loss_fn is not None and self.target_score is not None:
            return self.loss_fn(self.target_score, critic_score)
        elif self.loss_fn is not None and self.target_score is None:
            return self.loss_fn(y_true, critic_score)

def neg_mean_loss(y_true, y_pred):
    return -tf.reduce_mean(y_pred)

def mean_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

def mean_diff(y_true, y_pred):
    return K.mean(y_pred - y_true)


def norm_diff(y_true, y_pred):
    return K.abs(K.mean(K.abs(y_true)) - K.mean(K.abs(y_pred)))


def neg_norm(y_true, y_pred):
    return -K.abs(y_pred)


def l1_loss(y_true, y_pred):
    y_true_flat = K.batch_flatten(y_true)
    y_pred_flat = K.batch_flatten(y_pred)
    return K.sum(K.abs(y_pred_flat - y_true_flat), axis=-1)


def l1_norm(y_true, y_pred):
    return K.mean(K.abs(y_pred), axis=-1)


def neg_l1_norm(_, y_pred):
    return -K.mean(K.abs(y_pred), axis=-1)

# from keras siamese tutorial
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                                (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


class BoundLoss(object):
    def __init__(self, y_true, loss_fn):
        self.y_true = y_true
        self.loss_fn = loss_fn

    def compute_loss(self, y_true, y_pred):
        return self.loss_fn(self.y_true, y_pred)
        

def compute_pixelwise_categorical_crossentropy( y_true, y_pred ):
    # assume both inputs are im_h x im_w x n_labels
    eps = np.finfo('float32').eps
    # roll h and w dimensions into batch
    y_true = np.maximum(eps, np.minimum(1. - eps, y_true))
    y_pred = np.maximum(eps, np.minimum(1. - eps, y_pred))
    ce = np.sum(-y_true * np.log(y_pred) + (1. - y_true) * np.log(1. - y_pred), axis=-1, keepdims=True)
    return ce

if __name__ == '__main__':
#    test_true = np.tile(np.asarray( [[ [0,0.9], [0.7,0.2]]] ),(3,1,1))
    test_true = np.round(np.random.rand( 3,2,2,5 ))
    print(test_true)
#    test_pred = np.tile( np.asarray( [[[0.9,0.2], [0.7,0.8]]] ),(3,1,1))
#    test_pred = np.random.rand(3,2,2,5)
    test_pred = test_true
    print(test_pred)
