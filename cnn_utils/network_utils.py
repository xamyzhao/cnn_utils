import numpy as np
def assert_have_weights_changed(model, train_fn, train_kwargs, max_n_weights=10):
    old_model_weights = []
    for l in model.layers:
        w = l.get_weights()
        if w is not None and len(w) > 0:
            old_model_weights.append(w[0])
        if len(old_model_weights) > max_n_weights:
            break

    ret_vals = train_fn(**train_kwargs)

    new_model_weights = []
    for l in model.layers:
        w = l.get_weights()
        if w is not None and len(w) > 0:
            new_model_weights.append(w[0])
        if len(new_model_weights) > max_n_weights:
            break

    # at least some of the weights should be different
    assert np.any([not np.all(ow == new_model_weights[wi]) for wi, ow in enumerate(old_model_weights)])
    return ret_vals