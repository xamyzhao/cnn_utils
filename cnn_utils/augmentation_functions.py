import numpy as np
import cv2
import math
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator, interp2d, RectBivariateSpline
from scipy.ndimage import map_coordinates


def augScale(I, points=None, scale_rand=None, obj_scale=1.0, target_scale=1.0, pad_value=None, border_color=(0, 0, 0)):
    if scale_rand is not None:
        scale_multiplier = scale_rand
    else:
        scale_multiplier = randScale(0.8, 1.2)

    if isinstance(scale_multiplier, tuple) or isinstance(scale_multiplier, list):
        scale = target_scale / obj_scale * np.asarray(scale_multiplier)
        I_new = cv2.resize(I, (0, 0), fx=scale[0], fy=scale[1])
    else:
        scale = target_scale / obj_scale * scale_multiplier
        I_new = cv2.resize(I, (0, 0), fx=scale, fy=scale)

    target_size = I.shape
    border_size = (target_size[0] - I_new.shape[0], target_size[1] - I_new.shape[1])
    if border_size[0] > 0:
        I_new = cv2.copyMakeBorder(I_new, int(math.floor(border_size[0] / 2.0)), \
                                   int(math.ceil(border_size[0] / 2.0)), 0, 0, \
                                   cv2.BORDER_CONSTANT, value=border_color)
    elif border_size[0] < 0:
        I_new = I_new[
            -int(math.floor(border_size[0] / 2.0)):I_new.shape[0] + int(math.ceil(border_size[0] / 2.0))]

    if border_size[1] > 0:
        I_new = cv2.copyMakeBorder(I_new, 0, 0, int(math.floor(border_size[1] / 2.0)), \
                                   int(math.ceil(border_size[1] / 2.0)), \
                                   cv2.BORDER_CONSTANT, value=border_color)
    elif border_size[1] < 0:
        I_new = I_new[:, -int(math.floor(border_size[1] / 2.0)): I_new.shape[1] + int(math.ceil(border_size[1] / 2.0))]

    if points is not None:
        points = points * scale
    return I_new, points


def augRotate(I, points=None, max_rot_degree=30.0, crop_size_x=None, crop_size_y=None, degree_rand=None,
              border_color=(0, 0, 0)):
    if crop_size_x is None:
        crop_size_x = I.shape[1]
    if crop_size_y is None:
        crop_size_y = I.shape[0]

    if degree_rand is not None:
        degree = degree_rand
    else:
        degree = randRot(max_rot_degree)

    h = I.shape[0]
    w = I.shape[1]

    center = ((w - 1.0) / 2.0, (h - 1.0) / 2.0)
    R = cv2.getRotationMatrix2D(center, degree, 1)
    I = cv2.warpAffine(I, R, (crop_size_x, crop_size_y), borderValue=border_color, borderMode=cv2.BORDER_CONSTANT)

    if points is not None:
        for i in xrange(points.shape[0]):
            points[i, :] = rotatePoint(points[i, :], R)

    return I, points, degree


def rotatePoint(p, R):
    x_new = R[0, 0] * p[0] + R[0, 1] * p[1] + R[0, 2]
    y_new = R[1, 0] * p[0] + R[1, 1] * p[1] + R[1, 2]
    return np.array((x_new, y_new))


def augShift(I, joints=None, shift_px=2., rand_shift=None, border_color=(0, 0, 0)):
    if rand_shift is not None:
        x_shift = rand_shift[0]
        y_shift = rand_shift[1]
    else:
        x_shift, y_shift = randShift(shift_px)

    T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    I = cv2.warpAffine(I, T, None, borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)

    if joints:
        joints[:, 0] += x_shift
        joints[:, 1] += y_shift

    return I, joints


def augCrop(I, pad_to_size, crop_to_size=None, crop_to_size_range=None, crop_center=None, border_color=(0, 0, 0)):
    # if the crop size is not defined
    if crop_to_size is None:
        if crop_to_size_range is None:
            crop_to_size = pad_to_size
        else:
            # assume we have a list of two tuples, the first one is the row range
            # and the second tuple is the cols range
            crop_to_size = [
                np.random.rand(1)[0] * \
                    (crop_to_size_range[0][1] - crop_to_size_range[0][0]) + crop_to_size_range[0][0],
                np.random.rand(1)[0] * \
                    (crop_to_size_range[1][1] - crop_to_size_range[1][0]) + crop_to_size_range[1][0]
            ]
    crop_to_size = np.asarray(crop_to_size, dtype=np.float32)

    if crop_center is None:
        # min and max rows and columns
        min_rc = crop_to_size / 2.
        max_rc = I.shape[:2] - crop_to_size / 2.
        crop_center = np.random.rand(2) * (max_rc - min_rc) + min_rc

    start_row = max(0, int(np.round(crop_center[0] - crop_to_size[0] / 2.)))
    start_col = max(0, int(np.round(crop_center[1] - crop_to_size[1] / 2.)))
    I = I[start_row : start_row + int(crop_to_size[0]), start_col : start_col + int(crop_to_size[1])]

    # pad to the output shape if necessary
    I = image_utils.pad_or_crop_to_shape(I, pad_to_size, border_color=border_color)

    return I, crop_to_size, crop_center


aug_spaces = [(cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR, [1]), (cv2.COLOR_BGR2YCR_CB, cv2.COLOR_YCR_CB2BGR, [0]),
              (None, None, None)]


def rand_colorspace():
    return aug_spaces[int(round(np.random.rand(1)[0] * len(aug_spaces) - 1))]


def augSaturation(I, aug_percent=0.2, aug_colorspace=None, aug_channels=None):
    if aug_colorspace is None:
        aug_colorspace = rand_colorspace()

    if aug_channels is None:
        aug_channels = rand_channels(aug_colorspace)  # int(round(np.random.rand(1)[0]*2))

    for chan in aug_channels:
        I = scaleImgChannel(I, aug_colorspace[0], aug_colorspace[1], chan, aug_percent)
    return I


def augBlur(I, max_blur_sigma=10.0):
    blur_sigma = int(np.random.rand(1)[0] * max_blur_sigma)
    if blur_sigma > 0:
        kernel_size = int(blur_sigma / 5.0) * 2 + 1
        I = cv2.GaussianBlur(I, (kernel_size, kernel_size), blur_sigma)
    return I


def rand_channels(colorspace):
    if colorspace[2] is None:
        # select 1-3 channels randomly from the 3 channels available
        return np.random.choice(range(3), int(1 + np.random.rand(1) * 2), replace=False)
    else:
        return colorspace[2]


def augNoise(I, max_noise_sigma=0.1):
    noise_sigma = abs((np.random.randn(1)[0] - 0.5) * 2 * max_noise_sigma)

    rand_space = aug_spaces[int(round(np.random.rand(1)[0] * len(aug_spaces) - 1))]
    chans = rand_channels()

    I = I.astype(np.float32)

    if rand_space[0] is not None:
        I = cv2.cvtColor(I, rand_space[0])

    noise = np.zeros(I.shape, dtype=np.float32)

    for chan in chans:
        noise_sigma = min(0.05 * (np.max(I[:, :, chan]) - np.min(I[:, :, chan])), noise_sigma)
        noise[:, :, chan] = np.multiply(np.random.randn(I.shape[0], I.shape[1]), noise_sigma)

    if rand_space[1] is not None:
        I = cv2.cvtColor(I, rand_space[1])

    I = np.clip(np.add(I, noise), 0, 1.0)
    return I


def scaleImgChannel(I,
                    fwd_color_space=cv2.COLOR_BGR2HSV,
                    bck_color_space=cv2.COLOR_HSV2BGR,
                    channel=None,
                    aug_percent=0.2):
    if np.max(I) > 1.0 and not I.dtype == np.float32:
        I = np.multiply(I.astype(np.float32), 1 / 255.0)
    I = I.astype(np.float32)
    if fwd_color_space is not None:
        I = cv2.cvtColor(I, fwd_color_space)
    s_scale = aug_percent
    # s_scale = randScale( 1.0 - aug_percent, 1.0 + aug_percent )
    I[:, :, channel] = np.multiply(I[:, :, channel], s_scale)
    if fwd_color_space is not None:
        I = cv2.cvtColor(I, bck_color_space)

    return I


def swapLeftRight(joints):
    right = [3, 4, 5, 9, 10, 11]
    left = [6, 7, 8, 12, 13, 14]

    for i in xrange(6):
        ri = right[i] - 1
        li = left[i] - 1
        temp_x = joints[ri, 0]
        temp_y = joints[ri, 1]
        joints[ri, :] = joints[li, :]
        joints[li, :] = np.array([temp_x, temp_y])

    return joints


def augFlip(I, joints=None, obj_pos=None, flip_rand=None):
    if flip_rand is not None:
        do_flip = flip_rand
    else:
        do_flip = randFlip()
    if (do_flip):
        I = np.fliplr(I)
        if obj_pos is not None:
            obj_pos[0] = I.shape[1] - 1 - obj_pos[0]
        if joints is not None:
            joints[:, 0] = I.shape[1] - 1 - joints[:, 0]
            joints = swapLeftRight(joints)
    return I, joints, obj_pos


def randScale(scale_max, scale_min):
    rnd = np.random.rand()
    return (scale_max - scale_min) * rnd + scale_min


def randRot(max_rot_degrees):
    return (np.random.rand() - 0.5) * 2 * max_rot_degrees


def randShift(shift_px):
    x_shift = int(shift_px * (np.random.rand() - 0.5))
    y_shift = int(shift_px * (np.random.rand() - 0.5))
    return x_shift, y_shift


def randFlip():
    return np.random.rand() < 0.5


def inRange(joints, I, in_range):
    minLoc = 2
    for i in xrange(n_joints):
        if (joints[i, 0] < minLoc or joints[i, 1] < minLoc or
                    joints[i, 0] >= I.shape[1] or joints[i, 1] >= I.shape[0]):
            in_range[i] = False
    return in_range


def augProjective(I, max_theta=[15., 15., 15.], scale=1., max_shear=0.2):
    img_shape = I.shape
    if not type(max_theta) == list:
        max_theta = np.asarray([max_theta] * 3)
    I_in = I.astype(np.float32)

    I_out = np.zeros(img_shape, dtype=np.float32)
    theta = np.reshape(np.random.rand(3), (3,)) * np.reshape(max_theta, (3,)) * 2.0 - np.reshape(max_theta, (3,))
    theta = theta * math.pi / 180.0
    h = img_shape[0]
    w = img_shape[1]
    s = 1.
    R_x = np.asarray([[1., 0., 0., 0.],
                      [0., s * np.cos(theta[0]), -s * np.sin(theta[0]), 0.],
                      [0., s * np.sin(theta[0]), s * np.cos(theta[0]), 0.],
                      [0., 0., 0., 1.]], dtype=np.float32)
    R_z = np.asarray([[s * np.cos(theta[2]), -s * np.sin(theta[2]), 0., 0.],
                      [s * np.sin(theta[2]), s * np.cos(theta[2]), 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]], dtype=np.float32)
    R_y = np.asarray([[s * np.cos(theta[1]), 0., s * np.sin(theta[1]), 0.],
                      [0., 1., 0., 0.],
                      [-s * np.sin(theta[1]), 0., s * np.cos(theta[1]), 0.],
                      [0., 0., 0., 1.],
                      ], dtype=np.float32)

    R = np.matmul(R_y, R_x)
    RT = np.matmul(R_z, R)

    #
    xv, yv, zv = np.meshgrid(np.linspace(-w / 2., w / 2., w), np.linspace(-h / 2., h / 2., h), 0.)

    xyz = np.concatenate([np.reshape(xv, (1, np.prod(xv.shape))),
                          np.reshape(yv, (1, np.prod(yv.shape))),
                          np.reshape(zv, (1, np.prod(zv.shape))),
                          np.ones((1, np.prod(xv.shape)))], axis=0)
    xyz_camera = np.matmul(RT, xyz)

    f_x = 1.
    f_y = f_x

    x_0 = 0
    y_0 = 0

    if max_shear is None:
        s = 1.
    else:
        s = np.random.rand(1) * 2 * max_shear - max_shear

    xy_im = xyz_camera
    for c in range(I.shape[-1]):
        # image is currently centered at (0, 0), let's move it back to the center of the frame
        center_adjustment = np.tile(np.asarray([[w / 2.], [h / 2.]]), (1, xy_im.shape[-1]))
        Vq = map_coordinates(I_in[:, :, c].transpose(),
                             xy_im[:2] + center_adjustment, cval=1.)
        I_out[:, :, c] = np.reshape(Vq, img_shape[:-1])
    return I_out, theta


#	I_in_flat = np.reshape( I_in, 
if __name__ == '__main__':
    I = cv2.imread('/home/xamyzhao/MTGVS/db_all_new/14500.jpg')
    I = cv2.resize(I, None, fx=0.25, fy=0.25)
    I_ap, _ = augProjective(I, max_theta=60., max_shear=None)
    cv2.imwrite('projtest.jpg', I_ap)
