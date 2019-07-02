import cv2

from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import numpy as np

def visualize_embedding(embeddings, ims, labels,
                        title,
                        ax=None, x_lims=None, y_lims=None):
	# make a new figure if none specified
	if ax is None:
		plt.figure(figsize=(10, 10))
		ax = plt.subplot(111)

	x_min, x_max = np.min(embeddings, 0), np.max(embeddings, 0)
	print('Min: {}, max: {}'.format(x_min, x_max))

	n_points = embeddings.shape[0]

	for i in range(n_points):
		try:
			# if labels are numbers
			ax.text(embeddings[i, 0], embeddings[i, 1], '{}'.format(round(labels[i], 2)),
			        color=plt.cm.Set1((labels[i] - np.min(labels)) / (np.max(labels) - np.min(labels)) * 10.),
			        fontdict={'weight': 'bold', 'size': 9})
		except:
			ax.text(embeddings[i, 0], embeddings[i, 1], '{}'.format(labels[i]),
			        color=plt.cm.Set1(i / 10.),
			        fontdict={'weight': 'bold', 'size': 9})

	# set lims if given as inputs
	if x_lims is not None:
		ax.set_xlim(tuple(x_lims))
	else:
		ax.set_xlim((x_min[0], x_max[0]))

	if y_lims is not None:
		ax.set_ylim(tuple(y_lims))
	else:
		ax.set_ylim((x_min[1], x_max[1]))

	if hasattr(offsetbox, 'AnnotationBbox') and ims is not None:
		# only print thumbnails with matplotlib > 1.0
		shown_images = np.array([[1., 1.]])  # just something big
		for i in range(n_points):
			shown_images = np.r_[shown_images, [embeddings[i, :2]]]
			if ims.shape[-1] == 3:
				im = ims[i][:, :, [2, 1, 0]] # reverse bgr from opencv
			else:
				im = ims[i][:, :, 0]  # get rid of last channel for grayscale
			if np.any(im.shape[:2] > 128):
				im = cv2.resize(im, (80, 80))

			imagebox = offsetbox.AnnotationBbox(
				offsetbox.OffsetImage(im),
				embeddings[i, :2])
			ax.add_artist(imagebox)
	ax.set_title(title)
	return ax, (x_min[0], x_max[0]), (x_min[1], x_max[1])


def plot_vectors(embeddings, vecs, normalize_vecs=True, ax=None):
	if ax is None:
		plt.figure(figsize=(10, 10))
		ax = plt.subplot(111)

	embeddings_norm = embeddings - np.min(embeddings, 0)
	scale = 1. / np.max(embeddings_norm, 0)
	embeddings_norm = embeddings_norm * scale

	vecs_norm = vecs / np.tile(np.linalg.norm(vecs, axis=1)[:, np.newaxis], (1, 2)) * 0.1

	for i in range(embeddings.shape[0]):
		ax.arrow(
			embeddings_norm[i, 0] - vecs_norm[i, 0] / 2.,
			embeddings_norm[i, 1] - vecs_norm[i, 1] / 2.,
			vecs_norm[i, 0], vecs_norm[i, 1])

	# # normalize to make display nicer
	#     x_lim = np.max(embeddings[:,0]) - np.min(embeddings[:,0])

	#     y_lim = np.max(embeddings[:,1]) - np.min(embeddings[:,1])
	#     max_range = max(x_lim, y_lim)

	#     max_vec_norm = np.max(np.linalg.norm(vecs, axis=1))

	#     if normalize_vecs:
	#         vecs = vecs / max_vec_norm * max_range / 10.
	#     ax.set_xlim((np.min(embeddings[:, 0]), np.max(embeddings[:, 0])))
	#     ax.set_ylim((np.min(embeddings[:, 1]), np.max(embeddings[:, 1])))
	#     print(vecs)
	#     for i in range(embeddings.shape[0]):
	#         ax.arrow(embeddings[i,0] - vecs[i,0]/2., embeddings[i,1] - vecs[i, 1]/2., vecs[i,0], vecs[i, 1], head_width=1)

	return ax


def plot_gaussian(mean, std, mean_im, ax=None):
	if ax is None:
		plt.figure(figsize=(10, 10))
		ax = plt.subplot(111)

	# plot mean
	ax.scatter(mean[0], mean[1], marker='x')

	# draw vertical std
	xs = [mean[0], mean[0]]
	ys = [mean[1] - std[1], mean[1] + std[1]]
	# vert_line = lines.Line2D(xs, ys)
	ax.plot(xs, ys, '--', linewidth=2)

	xs = [mean[0] - std[0], mean[0] + std[0]]
	ys = [mean[1], mean[1]]
	ax.plot(xs, ys, '--', linewidth=2)

# horiz_line = lines.Line2D(xs, ys)
# ax.lines.extend([vert_line, horiz_line])
