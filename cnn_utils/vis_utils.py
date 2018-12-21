'''
Functions for visualizing matrices, especially batched matrices
'''

import numpy as np
import cv2

import textwrap
from cnn_utils import image_utils
import PIL
from PIL import Image, ImageDraw, ImageFont
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.misc as spm


def label_ims(ims_batch, labels=None,
              inverse_normalize=False,
              normalize=False,
              clip_flow=10, display_h=128, pad_top=None, clip_norm=None,
              color_space='rgb', combine_from_axis=0, concat_axis=0):
	'''
	Displays a batch of matrices as an image.

	:param ims_batch: n_batches x h x w x c array of images.
	:param labels: optional labels. Can be an n_batches length list of tuples, floats or strings
	:param inverse_normalize: boolean to do normalization from [-1, 1] to [0, 255]
	:param normalize: boolean to normalize any [min, max] to [0, 255]
	:param clip_flow: float for the min, max absolute flow magnitude to display
	:param display_h: integer number of pixels for the height of each image to display
	:param pad_top: integer number of pixels to pad each image at the top with (for more readable labels)
	:param color_space: string of either 'rgb' or 'ycbcr' to do color space conversion before displaying
	:param concat_axis: integer axis number to concatenate batch along (default is 0 for rows)

	:return:
	'''

	if len(ims_batch.shape) == 3 and ims_batch.shape[-1] == 3:
		# already an image
		return ims_batch

	# transpose the image until batches are in the 0th axis
	if not combine_from_axis == 0:
		# compute all remaining axes
		all_axes = range(len(ims_batch.shape))
		all_axes[combine_from_axis] = []

		ims_batch = np.transpose(ims_batch, (combine_from_axis,) + tuple(all_axes))

	batch_size = ims_batch.shape[0]
	h = ims_batch.shape[1]
	w = ims_batch.shape[2]
	if len(ims_batch.shape) == 3:
		n_chans = 1
	else:
		n_chans = ims_batch.shape[-1]

	if type(labels) == list and len(labels) == 1:  # only label the first image
		labels = labels + [''] * (batch_size - 1)
	elif labels is not None and not type(labels) == list and not type(labels) == np.ndarray:
		labels = [labels] * batch_size

	scale_factor = display_h / float(h)
	if pad_top:
		im_h = int(display_h + pad_top)

	else:
		im_h = display_h

	if len(ims_batch.shape) < 4:
		ims_batch = np.expand_dims(ims_batch, 3)

	if ims_batch.shape[-1] == 2:  # assume to be x,y flow; map to color im
		X_fullcolor = np.concatenate([ims_batch.copy(), np.zeros(ims_batch.shape[:-1] + (1,))], axis=3)

		if labels is not None:
			labels = [''] * batch_size

		for i in range(batch_size):
			X_fullcolor[i], min_flow, max_flow = flow_to_im(ims_batch[i], clip_flow=clip_flow)

			# also include the min and max flow in  the label
			if labels[i] is not None:
				labels[i] = '{},'.format(labels[i])
			else:
				labels[i] = ''

			for c in range(len(min_flow)):
				labels[i] += '({}, {})'.format(round(min_flow[c], 1), round(max_flow[c], 1))
		ims_batch = X_fullcolor.copy()

	elif inverse_normalize:
		ims_batch = image_utils.inverse_normalize(ims_batch)

	elif normalize:
		flattened_dims = np.prod(ims_batch.shape[1:])

		X_spatially_flat = np.reshape(ims_batch, (batch_size, -1, n_chans))
		X_orig_min = np.min(X_spatially_flat, axis=1)
		X_orig_max = np.max(X_spatially_flat, axis=1)

		# now actually flatten and normalize across channels
		X_flat = np.reshape(ims_batch, (batch_size, -1))
		if clip_norm is None:
			X_flat = X_flat - np.tile(np.min(X_flat, axis=1, keepdims=True), (1, flattened_dims))
			# avoid dividing by 0
			X_flat = X_flat / np.clip(np.tile(np.max(X_flat, axis=1, keepdims=True), (1, flattened_dims)), 1e-5, None)
		else:
			X_flat = X_flat - (-float(clip_norm))
			# avoid dividing by 0
			X_flat = X_flat / (2. * clip_norm)
			#X_flat = X_flat - np.tile(np.min(X_flat, axis=1, keepdims=True), (1, flattened_dims))
			# avoid dividing by 0
			#X_flat = X_flat / np.clip(np.tile(np.max(X_flat, axis=1, keepdims=True), (1, flattened_dims)), 1e-5, None)
			
		ims_batch = np.reshape(X_flat, ims_batch.shape)
		ims_batch = np.clip(ims_batch.astype(np.float32), 0., 1.)
		for i in range(batch_size):
			if labels is not None and len(labels) > 0:
				if labels[i] is not None:
					labels[i] = '{},'.format(labels[i])
				else:
					labels[i] = ''
				# show the min, max of each channel
				for c in range(n_chans):
					labels[i] += '({:.2f}, {:.2f})'.format(round(X_orig_min[i, c], 2), round(X_orig_max[i, c], 2))
	else:
		ims_batch = np.clip(ims_batch, 0., 1.)

	if color_space == 'ycbcr':
		for i in range(batch_size):
			ims_batch[i] = cv2.cvtColor(ims_batch[i], cv2.COLOR_YCR_CB2BGR)

	if np.max(ims_batch) <= 1.0:
		ims_batch = ims_batch * 255.0

	out_im = []
	for i in range(batch_size):
		# convert grayscale to rgb if needed
		if len(ims_batch[i].shape) == 2:
			curr_im = np.tile(np.expand_dims(ims_batch[i], axis=-1), (1, 1, 3))
		elif ims_batch.shape[-1] == 1:
			curr_im = np.tile(ims_batch[i], (1, 1, 3))
		else:
			curr_im = ims_batch[i]

		# scale to specified display size
		if scale_factor > 2:  # if we are upsampling by a lot, nearest neighbor can look really noisy
			interp = cv2.INTER_NEAREST
		else:
			interp = cv2.INTER_LINEAR

		if not scale_factor == 1:
			curr_im = cv2.resize(curr_im, None, fx=scale_factor, fy=scale_factor, interpolation=interp)

		if pad_top:
			curr_im = np.concatenate([np.zeros((pad_top, curr_im.shape[1], curr_im.shape[2])), curr_im], axis=0)
		out_im.append(curr_im)

	out_im = np.concatenate(out_im, axis=concat_axis).astype(np.uint8)
	font_size = 15
	max_text_width = int(17 * display_h / 128.)  # empirically determined
	if labels is not None and len(labels) > 0:
		im_pil = Image.fromarray(out_im)
		draw = ImageDraw.Draw(im_pil)
		for i in range(batch_size):
			if len(labels) > i:  # if we have a label for this image
				if type(labels[i]) == tuple or type(labels[i]) == list:
					# format tuple or list nicely
					formatted_text = ', '.join([
						labels[i][j].decode('UTF-8') if type(labels[i][j]) == np.unicode_ \
							else labels[i][j] if type(labels[i][j]) == str \
							else str(round(labels[i][j], 2)) if isinstance(labels[i][j], float) \
							else str(labels[i][j]) for j in range(len(labels[i]))])
				elif type(labels[i]) == float or type(labels[i]) == np.float32:
					formatted_text = str(round(labels[i], 2))  # round floats to 2 digits
				elif isinstance(labels[i], np.ndarray):
					# assume that this is a 1D array
					labels[i] = np.squeeze(labels[i])
					formatted_text = ', '.join([str(round(labels[i][j], 2)) for j in range(labels[i].size)])
				else:
					formatted_text = '{}'.format(labels[i])

				font = ImageFont.truetype('Ubuntu-M.ttf', font_size)
				# wrap the text so it fits
				formatted_text = textwrap.wrap(formatted_text, width=max_text_width)

				if display_h > 30:  # only print label if we have room
					for li, line in enumerate(formatted_text):
						if concat_axis == 0:
							draw.text((5, i * im_h + 5 + 14 * li), line, font=font, fill=(50, 50, 255))
						else:
							draw.text((5 + i * im_h, 5 + 14 * li), line, font=font, fill=(50, 50, 255))

		out_im = np.asarray(im_pil)
	
	return out_im


def pad_images_to_size(ims_list, pad_to_im_idx=None, ignore_axes=None, pad_val=0.):
	if pad_to_im_idx is not None:
		pad_to_shape = ims_list[pad_to_im_idx].shape
	else:
		im_shapes = np.reshape([im.shape for im in ims_list], (len(ims_list), -1))
		pad_to_shape = np.max(im_shapes, axis=0).tolist()

	if ignore_axes is not None:
		if not isinstance(ignore_axes, list):
			ignore_axes = [ignore_axes]
		for a in ignore_axes:
			pad_to_shape[a] = None

	ims_list = [image_utils.pad_or_crop_to_shape(im, pad_to_shape, border_color=pad_val) \
		for i, im in enumerate(ims_list)]
	return ims_list


def flow_to_im(flow, clip_flow=None):
	out_flow = np.zeros(flow.shape[:-1] + (3,))

	n_chans = flow.shape[-1]

	min_flow = [None] * n_chans
	max_flow = [None] * n_chans

	for c in range(n_chans):
		curr_flow = flow[:, :, c]
		if clip_flow is None:
			flow_vals = np.sort(curr_flow.flatten(), axis=None)
			min_flow[c] = flow_vals[int(0.01 * len(flow_vals))]
			max_flow[c] = flow_vals[int(0.99 * len(flow_vals))]
		else:
			min_flow[c] = -clip_flow
			max_flow[c] = clip_flow

		curr_flow = (curr_flow - min_flow[c]) * 1. / (max_flow[c] - min_flow[c])
		curr_flow = np.clip(curr_flow, 0., 1.)
		out_flow[:, :, c] = curr_flow * 255
	# flow = np.concatenate( [ flow, np.zeros( flow.shape[:-1]+(1,) ) ], axis=-1 ) * 255
	return out_flow, min_flow, max_flow


def xy_flow_to_im_cmap(flow):
	# assume flow is h x w x 2
	n_vals = 256
	cm = make_cmap_rainbow(n_vals)

	flow_mag = np.sqrt(np.sum(flow ** 2, axis=-1))
	flow_mag /= np.max(flow_mag)
	flow_angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
	flow_angle /= np.max(flow_angle)
	flow_angle *= (n_vals - 1)

	# put the angle into bins
	flow_angle_binned = np.digitize(flow_angle, np.linspace(0, n_vals-1, n_vals), right=True)
	print(np.max(flow_angle_binned))

	flow_im = cm[flow_angle_binned]

	flow_im_hsv = cv2.cvtColor(flow_im, cv2.COLOR_RGB2HSV)
	flow_im_hsv[:, :, 1] = flow_mag

	flow_im = cv2.cvtColor(flow_im_hsv, cv2.COLOR_HSV2RGB)
	return flow_im


# def overlay_labels_on_im( im, labels):
# assume labels is batch_size x h x w x n_labels
def make_cmap_rainbow_shuffled(nb_labels=256):
	hue = np.expand_dims(np.linspace(0, 1.0, nb_labels), 1).astype(np.float32)
	colors = np.concatenate([hue, np.ones((nb_labels, 2), dtype=np.float32)], axis=1) * 255
	colors = cv2.cvtColor(np.expand_dims(colors, 0).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)[0] / 255.0
	np.random.seed(17)
	color_idxs = np.random.permutation(nb_labels)
	print(color_idxs)
	colors = colors[color_idxs, :]
	#	print(colors.shape)
	return colors


def make_cmap_gradient(nb_labels=256, hue=1.0):
	hue = hue * np.ones((nb_labels, 1))
	sat = np.reshape(np.linspace(1., 0., nb_labels, endpoint=True), hue.shape)
	colors = np.concatenate([hue, sat, np.ones((nb_labels, 1), dtype=np.float32)], axis=1) * 255
	colors = cv2.cvtColor(np.expand_dims(colors, 0).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)[0] / 255.0
	return colors


def make_cmap_rainbow(nb_labels=256):
	hue = np.expand_dims(np.linspace(0, 1.0, nb_labels), 1).astype(np.float32)
	colors = np.concatenate([hue, np.ones((nb_labels, 2), dtype=np.float32)], axis=1) * 255
	colors = cv2.cvtColor(np.expand_dims(colors, 0).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)[0] / 255.0
	return colors


def plot_data_to_im(x, y, min_h=128):
	fig, ax = plt.subplots()
	ax.plot(x, y)
	ax.tick_params(axis='both', which='major', labelsize=14)
	fig.canvas.draw()
	X = np.array(fig.canvas.renderer._renderer)
	sf = float(min_h) / X.shape[0]
	X = spm.imresize(X[:, :, :3], float(sf))
	return X

# edited from adrian
def create_bw_grid(vol_size, spacing):
	"""
	create a black and white grid (white lines on a black volume)
	useful for visualizing a warp.
	"""
	grid_image = np.zeros(vol_size)
	grid_image[np.arange(0, vol_size[0], spacing, dtype=int), :] = 1
	grid_image[:, np.arange(0, vol_size[1], spacing, dtype=int)] = 1

	return grid_image

def flow_to_grid(flow, spacing=10):
	from scipy import interpolate
	grid = create_bw_grid(flow.shape[:2], spacing=spacing)
	h, w, c = flow.shape
	x = np.linspace(0, w, w, endpoint=False)
	y = np.linspace(0, h, h, endpoint=False)
	xx, yy = np.meshgrid(x, y)

	sample = flow[..., :2] + np.concatenate([np.expand_dims(xx, axis=-1), np.expand_dims(yy, axis=-1)], axis=-1)

	grid_warped = interpolate.interpn((y, x), grid, sample, bounds_error=False, fill_value=0)
	return np.reshape(grid_warped, grid.shape), grid

if __name__ == '__main__':
	warped = flow_to_grid(np.zeros((100, 100, 2)))
