import math

import cv2
import numpy as np
from scipy.ndimage import morphology as spndm
import scipy.interpolate as spi


def pad_or_crop_to_shape(
        I,
        out_shape,
        border_color=(255, 255, 255)):

    if not isinstance(border_color, tuple):
        n_chans = I.shape[-1]
        border_color = tuple([border_color] * n_chans)

    # an out_shape with a dimension value of None means just don't crop or pad in that dim
    border_size = [out_shape[d] - I.shape[d] if out_shape[d] is not None else 0 for d in range(2)]
    #border_size = (out_shape[0] - I.shape[0], out_shape[1] - I.shape[1])
    if border_size[0] > 0:
        I = cv2.copyMakeBorder(I,
                               int(math.floor(border_size[0] / 2.0)),
                               int(math.ceil(border_size[0] / 2.0)), 0, 0,
                               cv2.BORDER_CONSTANT, value=border_color)
    elif border_size[0] < 0:
        I = I[-int(math.floor(border_size[0] / 2.0)): I.shape[0]
                                                      + int(math.ceil(border_size[0] / 2.0)), :, :]
    if border_size[1] > 0:
        I = cv2.copyMakeBorder(I, 0, 0,
                               int(math.floor(border_size[1] / 2.0)),
                               int(math.ceil(border_size[1] / 2.0)),
                               cv2.BORDER_CONSTANT, value=border_color)
    elif border_size[1] < 0:
        I = I[:, -int(math.floor(border_size[1] / 2.0)): I.shape[1]
                                                         + int(math.ceil(border_size[1] / 2.0)), :]

    return I


def normalize(X):
    if X is None:
        return None
    return np.clip(X * 2.0 - 1.0, -1., 1.)


def inverse_normalize(X):
    return np.clip((X + 1.0) * 0.5, 0., 1.)


def seg_to_bounds(seg, delete_bound=5, save_raw_bounds_to_file=None, dilate_size=5, blur_size=5):
    seg = np.mean(seg, axis=-1)  # grayscale
    bounds = segutils.seg2contour(seg, contour_type='both')
    bounds[bounds > 0] = 1

    # delete any boundaries that are too close to the edge
    bounds[:delete_bound, :] = 0
    bounds[-delete_bound:, :] = 0
    bounds[:, :delete_bound] = 0
    bounds[:, -delete_bound:] = 0

    if save_raw_bounds_to_file:
        cv2.imwrite(save_raw_bounds_to_file, bounds * 255)

    if dilate_size > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        bounds = cv2.dilate(bounds.astype(np.float32), dilate_kernel)

    if blur_size > 0:
        bounds = cv2.GaussianBlur(bounds, ksize=(0, 0), sigmaX=blur_size, sigmaY=blur_size)
    if np.max(bounds) > 0:
        bounds = bounds / np.max(bounds)
    return bounds


def get_segmentation_mask(I, mask_color=(1., 1., 1.)):
    channel_masks = I.copy()
    for c in range(3):
        channel_masks[:, :, c] = (I[:, :, c] == mask_color[c]).astype(int)
    mask = np.prod(channel_masks, axis=-1)
    k = np.ones((3, 3), dtype=np.float32)
    mask = spndm.grey_closing(mask, footprint=k)
    mask = spndm.grey_opening(mask, footprint=k)
    mask = np.clip(mask, 0., 1.)
    return mask


def create_gaussian_kernel(sigma, n_sigmas_per_side=8, n_dims=2):
    t = np.linspace(-sigma * n_sigmas_per_side / 2, sigma * n_sigmas_per_side / 2, int(sigma * n_sigmas_per_side + 1))
    gauss_kernel_1d = np.exp(-0.5 * (t / sigma) ** 2)

    if n_dims == 2:
        gauss_kernel_2d = gauss_kernel_1d[:, np.newaxis] * gauss_kernel_1d[np.newaxis, :]
    else:
        gauss_kernel_2d = gauss_kernel_1d[:, np.newaxis, np.newaxis] * gauss_kernel_1d[np.newaxis, np.newaxis,
                                                                       :] * gauss_kernel_1d[np.newaxis, :, np.newaxis]
    gauss_kernel_2d = gauss_kernel_2d / np.sum(gauss_kernel_2d)

    #    cv2.imwrite('gauss_x.jpg', gauss_kernel_2d[gauss_kernel_2d.shape[0]/2, :, :]*255)
    #    cv2.imwrite('gauss_y.jpg', gauss_kernel_2d[:,gauss_kernel_2d.shape[1]/2, :, ]*255)
    #    cv2.imwrite('gauss_z.jpg', gauss_kernel_2d[:, :,gauss_kernel_2d.shape[2]/2 ]*255)
    # gauss_kernel_2d = np.reshape(gauss_kernel_2d, gauss_kernel_2d.shape + (1,1))
    return gauss_kernel_2d


