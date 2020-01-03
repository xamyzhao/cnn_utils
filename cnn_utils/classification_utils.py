import numpy as np


def is_onehot(labels):
    return type(labels) == np.ndarray and np.array_equal(np.max(labels, axis=1), np.ones((labels.shape[0],)))


def is_labels(oh):
    # if it's not an int, it's probably labels
    return (len(oh.shape) == 1 or oh.shape[-1] == 1) and ((oh.dtype == int and np.max(oh) > 1) or not oh.dtype == int)


def labels_to_onehot(labels, n_classes=0, label_mapping=None):
    if labels is None:
        return labels
    # we can use either n_classes (which assumes labels from 0 to n_classes-1) or a label_mapping
    if label_mapping is None and n_classes == 0:
        label_mapping = list(np.unique(labels))
        n_classes = len(np.unique(labels))  # np.max(labels)
    elif n_classes > 0 and label_mapping is None:
        # infer label mapping from # of classes
        label_mapping = np.linspace(0, n_classes, n_classes, endpoint=False).astype(int).tolist()
    elif n_classes == 0 and label_mapping is not None:
        n_classes = len(label_mapping)

    if isinstance(labels, np.ndarray) and labels.shape[-1] == len(label_mapping) and np.max(labels) <= 1. and np.min(labels) >= 0.:
        # already onehot
        return labels

    if isinstance(labels, np.ndarray) and labels.shape[-1] == 1:
        labels = np.take(labels, 0, axis=-1)

    labels = np.asarray(labels)

    if len(label_mapping) == 2 and 0 in label_mapping and 1 in label_mapping and type(
            labels) == np.ndarray and np.array_equal(np.max(labels, axis=-1), np.ones((labels.shape[0],))):
        return labels

    labels_flat = labels.flatten()
    onehot_flat = np.zeros(labels_flat.shape + (n_classes,), dtype=int)
    for li in range(n_classes):
        onehot_flat[np.where(labels_flat == label_mapping[li]), li] = 1

    onehot = np.reshape(onehot_flat, labels.shape + (n_classes,)).astype(np.float32)
    return onehot


def onehot_to_labels(oh, n_classes=0, label_mapping=None):
    # assume oh is batch_size (x R x C) x n_labels
    if n_classes > 0 and label_mapping is None:
        label_mapping = np.arange(0, n_classes)
    elif n_classes == 0 and label_mapping is None:
        label_mapping = list(np.arange(0, oh.shape[-1]).astype(int))
        n_classes = oh.shape[-1]
    elif n_classes == 0:
        n_classes = oh.shape[-1]

    argmax_idxs = np.argmax(oh, axis=-1).astype(int)
    labels = np.reshape(np.asarray(label_mapping)[argmax_idxs.flatten()], oh.shape[:-1]).astype(type(label_mapping[0]))

    return labels


def onehot_to_top_k_labels(oh, k, n_classes=None, label_mapping=None):
    # assume oh is batch_size (x R x C) x n_labels
    if n_classes > 0 and label_mapping is None:
        label_mapping = np.arange(0, n_classes)
    elif n_classes == 0 and label_mapping is None:
        label_mapping = list(np.arange(0, oh.shape[-1]).astype(int))
        n_classes = oh.shape[-1]
    elif n_classes == 0:
        n_classes = oh.shape[-1]
    oh_flat = np.reshape(oh, (np.prod(oh.shape[:-1]), n_classes))

    if type(label_mapping[0]) == str:
        labels_flat = np.empty((np.prod(oh.shape[:-1]), k), dtype='object')
        labels_flat[:] = ''
    else:
        labels_flat = np.zeros((np.prod(oh.shape[:-1]),), dtype=int)
    for i in range(oh_flat.shape[0]):
        max_vals, max_idxs = zip(*sorted(zip(oh_flat[i, :].tolist(),
                                             np.linspace(0, oh_flat.shape[1], oh_flat.shape[1], endpoint=False,
                                                         dtype=int).tolist())))
        argmax_idxs = max_idxs[:k]

        labels_flat[i] = label_mapping[argmax_idxs]
    labels = np.reshape(labels_flat, oh.shape[:-1] + (k,))
    return labels


def _test_onehot_to_labels():
    labels = ['bird', 'cat', 'dog', 'dog']
    label_mapping = ['ant', 'bird', 'cat', 'dog']
    oh = labels_to_onehot(np.asarray(labels), label_mapping=label_mapping)
    print(oh)
    labels_converted = onehot_to_labels(oh, label_mapping=label_mapping)
    print(labels_converted)
    assert np.array_equal(np.asarray(labels), labels_converted)
