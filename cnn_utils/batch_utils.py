import cv2
import numpy as np

from cnn_utils import classification_utils, image_utils, aug_utils


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
    if not isinstance(scale_factor, tuple):
        scale_factor = (scale_factor, scale_factor)

    n = X.shape[0]
    h = X.shape[1]
    w = X.shape[2]
    c = X.shape[3]

    max_chans = 100
    X_temp = np.transpose(X, (1, 2, 3, 0))
    X_temp = np.reshape(X_temp, (h, w, c * n))
    X_resized = []
    # do this in batches
    for bi in range(int(np.ceil(c * n / float(max_chans)))):
        X_batch = X_temp[..., bi * max_chans : min((bi + 1) * max_chans, X_temp.shape[-1])]
        n_batch_chans = X_batch.shape[-1]

        if np.max(X_batch) <= 1.0:
            X_batch = cv2.resize(X_batch * 255, None, fx=scale_factor[0], fy=scale_factor[1]) / 255.
        else:
            X_batch = cv2.resize(X_batch, None, fx=scale_factor[0], fy=scale_factor[1])
        X_resized.append(np.reshape(X_batch, X_batch.shape[:2] + (n_batch_chans,)))
    X_temp = np.concatenate(X_resized, axis=-1)
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
              batch_size, randomize=False,
              pad_or_crop_to_size=None, normalize_tanh=False,
              convert_onehot=False, labels_to_onehot_mapping=None,
              aug_model=None, aug_params=None,
              yield_aug_params=False, yield_idxs=False,
              random_seed=None):
    '''

    :param ims_data: list of images, or an image.
    If a single image, it will be automatically converted to a list

    :param labels_data: list of other data (e.g. labels) that do not require
    image normalization or augmentation, but might need to be converted to onehot

    :param batch_size:
    :param randomize: bool to randomize indices per batch

    :param pad_or_crop_to_size: pad or crop each image in ims_data to the specified size. Default pad value is 0
    :param normalize_tanh: normalize image to range [-1, 1], good for synthesis with a tanh activation
    :param aug_params:

    :param convert_onehot: convert labels to a onehot representation using the mapping below
    :param labels_to_onehot_mapping: list of labels e.g. [0, 3, 5] indicating the mapping of label values to channel indices

    :param yield_aug_params: include the random augmentation params used on the batch in the return values
    :param yield_idxs: include the indices that comprise this batch in the return values
    :param random_seed:
    :return:
    '''
    if random_seed:
        np.random.seed(random_seed)

    # make sure everything is a list
    if not isinstance(ims_data, list):
        ims_data = [ims_data]

    if not isinstance(normalize_tanh, list):
        normalize_tanh = [normalize_tanh] * len(ims_data)
    else:
        assert len(normalize_tanh) == len(ims_data)

    if aug_params is not None:
        if not isinstance(aug_params, list):
            aug_params = [aug_params] * len(ims_data)
        else:
            assert len(aug_params) == len(ims_data)
        out_aug_params = aug_params[:]

    if pad_or_crop_to_size is not None:
        if not isinstance(pad_or_crop_to_size, list):
            pad_or_crop_to_size = [pad_or_crop_to_size] * len(ims_data)
        else:
            assert len(pad_or_crop_to_size) == len(ims_data)


    # if we have labels that we want to generate from,
    # put everything into a list for consistency
    # (useful if we have labels and aux data)
    if labels_data is not None:
        if not isinstance(labels_data, list):
            labels_data = [labels_data]

        # each entry should correspond to an entry in labels_data
        if not isinstance(convert_onehot, list):
            convert_onehot = [convert_onehot] * len(labels_data)
        else:
            assert len(convert_onehot) == len(labels_data)


    idxs = [-1]

    n_ims = ims_data[0].shape[0]
    h = ims_data[0].shape[1]
    w = ims_data[0].shape[2]

    if pad_or_crop_to_size is not None:
        # pad each image and then re-concatenate
        ims_data = [np.concatenate([
            image_utils.pad_or_crop_to_shape(x, pad_or_crop_to_size)[np.newaxis]
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
        for i, im_data in enumerate(ims_data):
            X_batch = im_data[idxs]

            if not X_batch.dtype == np.float32 and not X_batch.dtype == np.float64:
                X_batch = X_batch.astype(np.float32) / 255.

            if normalize_tanh[i]:
                X_batch = image_utils.normalize(X_batch)

            if aug_params is not None and aug_params[i] is not None:
                if aug_model is not None:
                    # use the gpu aug model instead
                    T, _ = aug_utils.aug_params_to_transform_matrices(
                        batch_size=X_batch.shape[0], add_last_row=True,
                        **aug_params[i]
                    )
                    X_batch = aug_model.predict([X_batch, T])
                    out_aug_params[i] = T
                else:
                    X_batch, out_aug_params[i] = aug_utils.aug_im_batch(X_batch, **aug_params[i])
            ims_batches.append(X_batch)

        if labels_data is not None:
            labels_batches = []
            for li, Y in enumerate(labels_data):
                if Y is None:
                    Y_batch = None
                else:
                    if convert_onehot[li]:
                        Y_batch = classification_utils.labels_to_onehot(
                            Y[idxs],
                            label_mapping=labels_to_onehot_mapping)
                    else:
                        if isinstance(Y, np.ndarray):
                            Y_batch = Y[idxs]
                        else: # in case it's a list
                            Y_batch = [Y[idx] for idx in idxs]
                labels_batches.append(Y_batch)
        else:
            labels_batches = None

        if not randomize and restart_idxs:
            idxs[-1] = -1

        if yield_aug_params and yield_idxs:
            yield tuple(ims_batches) +  tuple(labels_batches) + (out_aug_params, idxs)
        elif yield_aug_params:
            yield tuple(ims_batches) + tuple(labels_batches) + (out_aug_params, )
        elif yield_idxs:
            yield tuple(ims_batches) + tuple(labels_batches) + (idxs, )
        else:
            yield tuple(ims_batches) + tuple(labels_batches)


def _test_gen_batch():
    '''
    Unit test for gen_batch
    :return:
    '''

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
