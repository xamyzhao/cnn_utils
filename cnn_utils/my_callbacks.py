import os
import time

import cv2
import keras.utils.generic_utils as keras_generic_utils
import tensorflow as tf
from keras import callbacks


class EveryNEpochs(callbacks.Callback):
    def __init__(self, call_every_n_epochs, do_fn, start_epoch=0, start_iter=0,
            input_epochs=True,
            input_iters=False):
        self.call_every_n_epochs = call_every_n_epochs

        self.epoch_count = start_epoch
        self.iter_count = start_iter

        self.input_iters = input_iters
        self.input_epochs = input_epochs
        self.do_fn = do_fn

    def on_train_begin(self,  logs={}):
        return 0

    def on_epoch_end(self, epoch, logs):
        self.epoch_count += 1
        if self.epoch_count % self.call_every_n_epochs == 0:
            if self.input_epochs and self.input_iters:
                self.do_fn(epoch=self.epoch_count, iter_count=self.iter_count)
            elif not self.input_epochs and self.input_iters:
                self.do_fn(iter_count=self.iter_count)
            elif self.input_epochs and not self.input_iters:
                self.do_fn(epoch=self.epoch_count)
            else:
                self.do_fn()

    def on_batch_end(self, batch, logs={}):
        self.iter_count += 1
        return 0


class PrintResults(callbacks.Callback):
    def __init__(self, make_im_fn, out_dir, file_prefix='train',
                       save_every_n_seconds=None,
                       save_every_n_epochs=None,
                        logger=None,
                    start_epoch=0,
                 ):


        self.save_every_n_batches = 0
        self.save_every_n_seconds = save_every_n_seconds
        self.save_every_n_epochs = save_every_n_epochs

        self.batch_count = 0
        self.iter_count = 0
        self.epoch_count = start_epoch

        self.out_dir = out_dir
        self.file_prefix = file_prefix
        self.make_im_fn = make_im_fn

        self.logger = logger

        self.start_time = time.time()

    def on_train_start(self, logs={}):
        return 0

    def on_epoch_end(self, epoch, logs):
        self.epoch_count += 1

        # reset batch count
        self.batch_count = 0
        if (self.save_every_n_epochs is not None and self.epoch_count % self.save_every_n_epochs == 0):
            self._save_results()

    def on_batch_end(self, batch, logs={}):
        self.batch_count += 1
        self.iter_count += 1

        # time how long it takes to do 10 batches
        if self.save_every_n_seconds is not None and self.iter_count == 10 and self.save_every_n_batches == 0:
            runtime = time.time() - self.start_time
            s_per_batch = runtime / 10.
            self.save_every_n_batches = max(1, int(round(self.save_every_n_seconds / s_per_batch)))
            if self.logger is not None:
                self.logger.debug('Printing results every {} batches'.format(self.save_every_n_batches))

        if (self.save_every_n_seconds is not None and self.save_every_n_batches > 0
                and self.iter_count % self.save_every_n_batches == 0):
            self._save_results()

    def _save_results(self):
        im = self.make_im_fn()
        out_file = os.path.join(self.out_dir, '_'.join([
            self.file_prefix,
            'epoch{}'.format(self.epoch_count),
            'batch{}'.format(self.batch_count)
        ]) + '.jpg')
        cv2.imwrite(out_file, im)


class LogLosses(callbacks.Callback):
    def __init__(self, print_fn,
                 log_every_n_batches=None, log_every_n_epochs=None,
                 loss_prefix=None,
                loss_names=None,
                start_epoch=0
         ):
        self.print_fn = print_fn
        self.loss_prefix = loss_prefix
        self.iter_count = 0
        self.epoch_count = start_epoch
        self.log_every_n_batches = log_every_n_batches
        self.log_every_n_epochs = log_every_n_epochs
        self.loss_names = loss_names

    def on_train_begin(self, logs={}):
        self.metric_names = self.params.get('metrics')

    def on_epoch_end(self, epoch, logs):
        self.epoch_count += 1
        if (self.log_every_n_epochs is not None and self.epoch_count % self.log_every_n_epochs == 0):
            self._log_losses(logs)

    def _log_losses(self, logs):
        if self.loss_prefix is not None and self.loss_names is None:
            metric_names = [self.loss_prefix + '_' + m for m in self.metric_names if m in logs]
        elif self.loss_names is None:
            metric_names = [m for m in self.metric_names if m in logs]
        else:
            metric_names = self.loss_names
        metrics = [logs.get(m) for m in self.metric_names if m in logs]
        self.print_fn(loss_names=metric_names, loss_vals=metrics,
                      iter_count=self.iter_count)

    def on_iter_end(self, iter, logs={}):
        self.iter_count += 1

        if (self.log_every_n_batches is not None
                and self.iter_count % self.log_every_n_batches == 0):
            self._log_losses(logs)


class TensorBoard_ScalarLosses(callbacks.Callback):
    def __init__(self, tensorboard_writer, loss_names=None, start_iter=0):
        self.tbw = tensorboard_writer
        self.iter_count = start_iter
        self.loss_names = loss_names

    def on_train_begin(self, logs={}):
        self.metric_names = self.params.get('metrics')

    def on_iter_end(self, iter, logs={}):
        self.iter_count += 1
        if self.loss_names is None:
            metric_names = [m for m in self.metric_names if m in logs]
        else:
            metric_names = self.loss_names
        metrics = [logs.get(m) for m in self.metric_names if m in logs]
        for i in range(len(metrics)):
            self.tbw.add_summary(tf.Summary(
                    value=[tf.Summary.Value(tag=metric_names[i], simple_value=metrics[i]), ]
                ), self.iter_count)
            #if i >= len(metrics):
            #	break

class ProgbarWrapper(callbacks.Callback):
    def __init__(self, loss_names, n_batch_per_epoch,
            model_name=None, start_epoch=0, end_epoch=None,
        ):
        self.loss_names = loss_names
        self.n_batch_per_epoch = n_batch_per_epoch
        self.progbar = keras_generic_utils.Progbar(self.n_batch_per_epoch)
        self.model_name = model_name
        self.end_epoch = end_epoch
        self.epoch_count = start_epoch

    def on_train_begin(self, logs={}):
        self.metric_names = self.params.get('metrics')

    def on_epoch_end(self, epoch, logs):
        # reset counter
        self.epoch_count += 1
        print('{} epoch {} of {}'.format(self.model_name, self.epoch_count, self.end_epoch))
        self.progbar = keras_generic_utils.Progbar(self.n_batch_per_epoch)

    def on_batch_end(self, batch, logs={}):
        if self.loss_names is None:
            loss_names = [m for m in self.metric_names if m in logs]
        else:
            loss_names = self.loss_names
        metrics = [logs.get(m) for m in self.metric_names if m in logs]
        self.progbar.add(1, values=[(loss_names[i], metrics[i]) for i in range(len(metrics))])

