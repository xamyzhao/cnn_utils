import functools
import logging
import os
import sys
import time

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.utils import generic_utils
from tensorflow.python.client import timeline

from cnn_utils import my_callbacks
import json


def configure_gpus(gpus):
    # set gpu id and tf settings
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpus])
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    K.tensorflow_backend.set_session(tf.Session(config=config))


# loads a saved experiment using the saved parameters.
# runs all initialization steps so that we can use the models right away
def load_experiment_from_dir(from_dir, exp_class,
                             load_n=None,
                             load_epoch=None,
                             do_logging=False,  # dont log if we are just loading this exp for evaluation
                             do_load_models=True,
                             prompt_update_name=True,
                             verbose=True,
                             ):
    with open(os.path.join(from_dir, 'arch_params.json'), 'r') as f:
        fromdir_arch_params = json.load(f)
        fromdir_arch_params['exp_dir'] = from_dir
    with open(os.path.join(from_dir, 'data_params.json'), 'r') as f:
        fromdir_data_params = json.load(f)

    exp = exp_class(
        data_params=fromdir_data_params, arch_params=fromdir_arch_params,
        loaded_from_dir=from_dir,
        prompt_delete_existing=False, # just continue in exactly the dir that was specified
        prompt_update_name=prompt_update_name, # in case the experiment was renamed
        do_logging=do_logging)

    exp.load_data(load_n=load_n)
    exp.create_models(verbose=verbose)

    if do_load_models:
        loaded_epoch = exp.load_models(load_epoch)
    else:
        loaded_epoch = None

    return exp, loaded_epoch

def run_experiment(exp, run_args,
                   end_epoch,
                   save_every_n_epochs, test_every_n_epochs,
                   early_stopping_eps=None):
    if run_args.debug:
        if run_args.epoch is not None:
            end_epoch = int(run_args.epoch) + 10
        else:
            end_epoch = 10

        # TODO: find a better way to set default params in case our experiment doesnt use them?
        if hasattr(run_args, 'loadn') and run_args.loadn is None:
            run_args.loadn = 1
        elif not hasattr(run_args, 'loadn'):
            run_args.loadn = None


        save_every_n_epochs = 2
        test_every_n_epochs = 2

        exp.set_debug_mode(True)

    if run_args.batch_size is None:
        run_args.batch_size = 8

    if not hasattr(run_args, 'ignore_missing'):
        run_args.ignore_missing = False

    exp_dir, figures_dir, logs_dir, models_dir = exp.get_dirs()

    # log to the newly created experiments dir
    formatter = logging.Formatter(
        '[%(asctime)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    lfh = logging.FileHandler(
        filename=os.path.join(exp_dir, 'training.log'))
    lsh = logging.StreamHandler(sys.stdout)
    lfh.setFormatter(formatter)
    lsh.setFormatter(formatter)
    lfh.setLevel(logging.DEBUG)
    lsh.setLevel(logging.DEBUG)

    file_stdout_logger = logging.getLogger('both')
    file_stdout_logger.setLevel(logging.DEBUG)
    file_stdout_logger.addHandler(lfh)
    file_stdout_logger.addHandler(lsh)

    file_logger = logging.getLogger('file')
    file_logger.setLevel(logging.DEBUG)
    file_logger.addHandler(lfh)

    # load the dataset. load fewer if debugging
    exp.load_data(load_n=run_args.loadn)

    # create models and load existing ones if necessary
    exp.create_models()

    start_epoch = exp.load_models(run_args.epoch,
        stop_on_missing=not run_args.ignore_missing,
        init_layers=run_args.init_weights)

    # compile models for training
    if run_args.do_profile:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None

    exp.compile_models(run_options=run_options, run_metadata=run_metadata)

    if run_args.init_from:
        exp.init_model_weights(run_args.init_from)

    exp.create_generators(batch_size=run_args.batch_size)

    tbw = tf.summary.FileWriter(logs_dir)
    if run_args.fit:
        train_using_fit_generator(
            exp=exp, batch_size=run_args.batch_size,
            start_epoch=start_epoch, end_epoch=end_epoch,
            save_every_n_epochs=save_every_n_epochs,
            test_every_n_epochs=test_every_n_epochs,
            tbw=tbw, file_stdout_logger=file_stdout_logger, file_logger=file_logger,
            run_args=run_args,
            early_stopping_eps=early_stopping_eps
        )
    else:
        train_batch_by_batch(
            exp=exp, batch_size=run_args.batch_size,
            start_epoch=start_epoch, end_epoch=end_epoch,
            save_every_n_epochs=save_every_n_epochs,
            test_every_n_epochs=test_every_n_epochs,
            tbw=tbw, file_stdout_logger=file_stdout_logger, file_logger=file_logger,
            run_args=run_args,
            early_stopping_eps=early_stopping_eps,
            run_metadata=run_metadata,
        )

    return exp_dir





def train_using_fit_generator(
        exp, batch_size,
        start_epoch, end_epoch,
        tbw, file_stdout_logger, file_logger,
        run_args, save_every_n_epochs, test_every_n_epochs
):
    def refresh_logs(epoch=None):  # arg is purely for EveryNEpochs callback
        file_stdout_logger.handlers[0].close()  # flush our .log file
        lfh = logging.FileHandler(filename=os.path.join(exp_dir, 'training.log'))
        formatter = logging.Formatter('[%(asctime)s] %(message)s', "%Y-%m-%d %H:%M:%S")
        lfh.setFormatter(formatter)
        lfh.setLevel(logging.DEBUG)
        file_stdout_logger.handlers[0] = lfh

        if exp.logger is not None:
            exp._reopen_log_file()

        tbw.close()  # save to disk and then open a new file so that we can read into tensorboard more easily
        tbw.reopen()

    def log_test_losses(iter_count):
        test_losses, test_loss_names = exp.test_joint()
        network_utils.log_losses(
            progressBar=None, tensorBoardWriter=tbw, logger=file_stdout_logger,
            loss_names=test_loss_names, loss_vals=test_losses, iter_count=iter_count)

    n_batch_per_epoch = min(run_args.mbpe, int(np.ceil(exp.get_n_train() / float(batch_size))))

    # initialize all callbacks here so they maintain their own epoch counts
    callbacks = [
        my_callbacks.ProgbarWrapper(loss_names=exp.loss_names, n_batch_per_epoch=n_batch_per_epoch,
                                    model_name=exp.get_model_name(), start_epoch=start_epoch, end_epoch=end_epoch,
                                    ),
        # keras_callbacks.ProgbarLogger(),
        my_callbacks.PrintResults(save_every_n_seconds=run_args.print_every,
                                  make_im_fn=exp.make_train_results_im,
                                  out_dir=figures_dir,
                                  file_prefix='train',
                                  start_epoch=start_epoch,
                                  logger=file_stdout_logger,
                                  ),
        my_callbacks.EveryNEpochs(  # make sure the exp keeps track of the epoch count
            call_every_n_epochs=1,
            start_epoch=start_epoch,
            do_fn=exp.update_epoch_count),

        # save models every so often. we don't use the keras function since we might want to save some modules too
        my_callbacks.EveryNEpochs(
            call_every_n_epochs=save_every_n_epochs,
            start_epoch=start_epoch,
            start_iter=start_epoch * n_batch_per_epoch,
            do_fn=functools.partial(exp.save_models, models_dir=models_dir),
            input_iters=True,
        ),

        # close and reopen log files so we can look at them
        my_callbacks.EveryNEpochs(
            call_every_n_epochs=save_every_n_epochs,
            start_epoch=start_epoch,
            do_fn=refresh_logs
        ),
        # call test function in case we are doing something interesting
        my_callbacks.EveryNEpochs(
            call_every_n_epochs=test_every_n_epochs,
            start_epoch=start_epoch,
            do_fn=log_test_losses,
            input_iters=True,
            input_epochs=False,
        ),

        # save test results
        my_callbacks.PrintResults(
            save_every_n_epochs=test_every_n_epochs,
            out_dir=figures_dir,
            file_prefix='test',
            make_im_fn=exp.make_test_results_im,
            logger=file_stdout_logger,
            start_epoch=start_epoch,
        ),

        # write training losses to and training.log file every epoch
        my_callbacks.LogLosses(
            print_fn=functools.partial(
                network_utils.log_losses,
                progressBar=None, tensorBoardWriter=tbw, logger=file_logger),
            loss_names=['train_' + ln for ln in exp.loss_names],  # + ['val_' + ln for ln in exp.loss_names],
            start_epoch=start_epoch,
            log_every_n_epochs=1,
        ),
        my_callbacks.TensorBoard_ScalarLosses(
            tbw,
            loss_names=['train_' + ln for ln in exp.loss_names],  # \
            #                 + ['val_' + ln for ln in exp.loss_names],
            start_iter=start_epoch * n_batch_per_epoch,
        ),
        # TODO: early stopping
    ]

    while exp.epoch_count < end_epoch:  # we might need to call fit_generator on different models throughout training
        # assumes that each experiment has a main trainer_model
        exp.trainer_model.fit_generator(
            exp.train_gen,
            steps_per_epoch=n_batch_per_epoch,
            epochs=end_epoch,
            verbose=0,
            callbacks=callbacks,
            # validation_data=exp.valid_gen,
            # validation_steps=10,#int(np.ceil(exp.get_n_test() / float(batch_size))),
            initial_epoch=start_epoch,
        )

        for c in callbacks:
            if isinstance(c, my_callbacks.LogLosses):
                c.loss_names = ['train_' + ln for ln in exp.loss_names] + ['val_' + ln for ln in exp.loss_names]
            elif hasattr(c, 'loss_names'):
                c.loss_names = exp.loss_names


def train_batch_by_batch(
        exp,
        batch_size,
        start_epoch, end_epoch, save_every_n_epochs, test_every_n_epochs,
        tbw, file_stdout_logger, file_logger,
        run_args,
        early_stopping_eps,
        run_metadata=None
):
    max_n_batch_per_epoch = 1000  # limits each epoch to batch_size * 1000 examples. i think this is ok.
    n_batch_per_epoch_train = min(max_n_batch_per_epoch, int(np.ceil(exp.get_n_train() / float(batch_size))))

    max_printed_examples = 8
    print_every = 100000  # set this to be really high at  first
    print_atleast_every = 100
    print_atmost = max(1, max_printed_examples / batch_size)


    # lets say we want 1 new result image every 1 minute
    print_every_n_seconds = run_args.print_every

    # save a new model every 20 minutes? seems reasonable
    auto_save_every_n_epochs = 100
    auto_test_every_n_epochs = 100
    min_save_every_n_epochs = 10
    save_every_n_seconds = 20 * 60

    #    print_n_batches_per_epoch = max(1, max_printed_examples / batch_size)
    # we don't want more than 64 images printed per epoch
    #    print_every = int(np.floor(
    #                    ((n_batch_per_epoch_train-1) / print_n_batches_per_epoch) / 2)) * 2 + 1  # make this odd so we can print augmentations

    start_time = time.time()

    # do this once here to flush any setup information to the file
    exp._reopen_log_file()

    for e in range(start_epoch, end_epoch + 1):
        file_stdout_logger.debug('{} training epoch {}/{}'.format(exp.model_name, e, end_epoch + 1))

        if e < end_epoch:
            exp.update_epoch_count(e)

        pb = generic_utils.Progbar(n_batch_per_epoch_train)
        printed_count = 0
        for bi in range(n_batch_per_epoch_train):
            disc_loss, disc_loss_names = exp.train_discriminator()
            joint_loss, joint_loss_names = exp.train_joint()
            batch_count = e * n_batch_per_epoch_train + bi

            # only log to file on the last batch of training, otherwise we'll have too many messages
            training_logger = None
            if bi == n_batch_per_epoch_train - 1:
                training_logger = file_logger

            log_losses(pb, tbw, training_logger,
                                     disc_loss_names + joint_loss_names,
                                     disc_loss + joint_loss,
                                     batch_count)

            # time how long it takes to do 5 batches
            if batch_count - start_epoch * n_batch_per_epoch_train == 5:
                s_per_batch = (time.time() - start_time) / 5.

                # make this an odd integer in case our experiment is doing
                # different things on alternating batches, so that we can visualize both
                print_every = int(np.ceil(print_every_n_seconds / s_per_batch / 2.)) * 2 + 1
                auto_save_every_n_epochs = save_every_n_seconds / s_per_batch / n_batch_per_epoch_train
                if auto_save_every_n_epochs > 50:  # if interval is big enough, adjust to multiples of 50
                    auto_save_every_n_epochs = max(1, int(np.floor(save_every_n_epochs / 50))) * 50
                else:
                    auto_save_every_n_epochs = max(1, int(np.floor(save_every_n_epochs / min_save_every_n_epochs))) \
                                               * min_save_every_n_epochs


            if ((batch_count % print_every == 0 or batch_count % print_atleast_every == 0)) \
                    and printed_count < print_atmost:
                results_im = exp.make_train_results_im()
                cv2.imwrite(
                    os.path.join(exp.figures_dir,
                                 'train_epoch{}_batch{}.jpg'.format(e, bi)
                                 ),
                    results_im)
                printed_count += 1

        if batch_count >= 10:  # TODO: make this only print once?
            file_stdout_logger.debug('Printing every {} batches, '
                                     'saving every {} and {} epochs, '
                                     'testing every {}'.format(print_every,
                                                               auto_save_every_n_epochs,
                                                               save_every_n_epochs,
                                                               test_every_n_epochs,
                                                               ))

        if run_args.do_profile:
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open(os.path.join(exp.exp_dir, 'tf_timeline.ctf.json'), 'w') as f:
                f.write(trace.generate_chrome_trace_format())

        if (e > 0 and e % auto_save_every_n_epochs == 0 and e > start_epoch) or e == end_epoch or (
                            e > 0 and e % save_every_n_epochs == 0 and e > start_epoch):
            exp.save_models(e, iter_count=e * n_batch_per_epoch_train)
            # TODO: figure out how to flush log file without closing
            file_stdout_logger.handlers[0].close()  # flush our .log file
            lfh = logging.FileHandler(filename=os.path.join(exp.exp_dir, 'training.log'))
            file_stdout_logger.handlers[0] = lfh

            if exp.logger is not None:
                exp.logger.handlers[0].close()
                exp._reopen_log_file()

            tbw.close()  # save to disk and then open a new file so that we can read into tensorboard more easily
            tbw.reopen()

        if (e % auto_test_every_n_epochs == 0 or e % test_every_n_epochs == 0):
            file_stdout_logger.debug('{} testing'.format(exp.model_name))
            pbt = generic_utils.Progbar(1)

            test_loss, test_loss_names = exp.test_joint()

            log_losses(pbt, None, file_logger,
                                     test_loss_names, test_loss,
                                     e * n_batch_per_epoch_train + bi)

            results_im = exp.make_test_results_im(e)
            if results_im is not None:
                cv2.imwrite(os.path.join(exp.figures_dir, 'test_epoch{}_batch{}.jpg'.format(e, bi)), results_im)

            log_losses(None, tbw, file_logger,
                                     test_loss_names, test_loss,
                                     e * n_batch_per_epoch_train + bi)

            if run_args.early_stopping:
                if not np.any(np.isnan(exp.validation_losses_buffer)) \
                        and len(exp.validation_losses_buffer) > 0 \
                        and np.all(exp.validation_losses_buffer[1:] - exp.validation_losses_buffer[
                            0] < early_stopping_eps):
                    file_stdout_logger.debug('Validation losses {}, stopping!'.format(exp.validation_losses_buffer))
                    sys.exit()
            print('\n\n')


def log_losses(progressBar, tensorBoardWriter, logger, loss_names, loss_vals, iter_count):
    if not isinstance(loss_vals, list):  # occurs when model only has one loss
        loss_vals = [loss_vals]

    # update the progress bar displayed in stdout
    if progressBar is not None:
        progressBar.add(1, values=[(loss_names[i], loss_vals[i]) for i in range(len(loss_vals))])

    # write to log using python logging
    if logger is not None:
        logger.debug(', '.join(['{}: {}'.format(loss_names[i], loss_vals[i]) for i in range(len(loss_vals))]))

    # write to tensorboard for pretty plots
    if tensorBoardWriter is not None:
        for i in range(len(loss_names)):
            tensorBoardWriter.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag=loss_names[i], simple_value=loss_vals[i]), ]), iter_count)
            if i >= len(loss_vals):
                break
