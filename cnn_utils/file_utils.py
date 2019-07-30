import json
import os
import re

import numpy as np


def filenames_to_im_ids(im_files):
    if isinstance(im_files[0], int):
        return im_files
    elif 'frame_' in im_files[0]:
        im_file_ids = [int(re.search('(?<=frame_)[0-9]*', f).group(0)) for f in im_files]
        return im_file_ids
    elif 'frame' in im_files[0]:
        im_file_ids = [int(re.search('(?<=frame)[0-9]*', os.path.basename(f)).group(0)) for f in im_files]
        return im_file_ids
    try:
        int(im_files[0])
        im_file_ids = [ int( os.path.splitext(os.path.basename(f))[0] ) for f in im_files ]
    
    except ValueError:
        '''
        # just return these as strings? not sure why i wanted to do the conversion
        im_file_ids = [os.path.splitext(os.path.basename(f))[0] for f in im_files]
        
        '''
        if '_' in im_files[0]:
            im_file_ids = [ int( os.path.splitext(os.path.basename(f))[0].split('_')[0] ) for f in im_files ]
        else:
            im_file_ids = im_files[:]
            for i,f in enumerate(im_files):
                curr_str = []
                first_seg = os.path.splitext(os.path.basename(f))[0].replace('_','') # TODO: check this for all naming formats we use
                for c in first_seg:
                    try:
                        curr_str.append( str(int(c)))
                    except ValueError:
                        curr_str.append( str(ord(c)))
                im_file_ids[i] = int(''.join(curr_str))
    #        im_file_ids = [ ''.join([str(ord(c)) for c in ]) for f in im_files ]
    #        print(im_file_ids)
    return im_file_ids


def _prompt_rename(old_dir, new_dir, debug_input=None):
    print('Rename dir \n{} to \n{} [y/N]?'.format(old_dir, new_dir))

    if debug_input:
        print('Debug input: {}'.format(debug_input))
        choice = debug_input
    else:
        choice = input().lower()

    rename_choices = ['yes', 'y', 'ye']
    keep_choices = ['no', 'n']

    if choice in rename_choices:
        return new_dir, True
    else:
        return old_dir, False


def make_output_dirs(experiment_base_name: str,
                     prompt_delete_existing: bool = True,
                     prompt_update_name: bool = True,
                     exp_root: str = './experiments/',
                     existing_exp_dir=None,
                     # for unit testing
                     debug_delete_input=None,
                     debug_rename_input=None
                     ):
    '''
    Creates the experiment directory (for storing log files, parameters) as well as subdirectories for
    files, logs and models.

    If a directory already exists for this experiment,

    :param experiment_base_name: desired name for the experiment
    :param prompt_delete_existing: if we find an existing directory with the same name,
        do we tell the user? if not, just continue in the existing directory by default
    :param prompt_update_name: if the new experiment name differs from the existing_exp_dir,
        do we try to rename the existing directory to the new naming scheme?
    :param exp_root: root directory for all experiments
    :param existing_exp_dir: previous directory (if any) of this experiment
    :return:
    '''

    do_rename = False

    if existing_exp_dir is None:
        # brand new experiment
        experiment_name = experiment_base_name
        target_exp_dir = os.path.join(exp_root, experiment_base_name)
    else:  # we are loading from an existing directory
        if re.search('_[0-9]*$', existing_exp_dir) is not None:
            # we are probably trying to load from a directory like experiments/<exp_name>_1,
            #  so we should track the experiment_name with the correct id
            experiment_name = os.path.basename(existing_exp_dir)
            target_exp_dir = os.path.join(exp_root, experiment_name)
        else:
            # we are trying to load from a directory, but the newly provided experiment name doesn't match.
            # this can happen when the naming scheme has changed
            target_exp_dir = os.path.join(exp_root, experiment_base_name)

            # if it has changed, we should prompt to rename the old experiment to the new one
            if prompt_update_name and not os.path.abspath(existing_exp_dir) == os.path.abspath(target_exp_dir):
                target_exp_dir, do_rename = _prompt_rename(
                    existing_exp_dir, target_exp_dir, debug_rename_input)

                if do_rename: # we might have changed the model name to something that exists, so prompt if so
                    print('Renaming {} to {}!'.format(existing_exp_dir, target_exp_dir))
                    prompt_delete_existing = True
            else:
                target_exp_dir = existing_exp_dir # just assume we want to continue in the old one

            experiment_name = os.path.basename(target_exp_dir)

    print('Existing exp dir: {}'.format(existing_exp_dir))
    print('Target exp dir: {}'.format(target_exp_dir))

    figures_dir = os.path.join(target_exp_dir, 'figures')
    logs_dir = os.path.join(target_exp_dir, 'logs')
    models_dir = os.path.join(target_exp_dir, 'models')

    copy_count = 0

    # check all existing dirs with the same prefix (and all suffixes e.g. _1, _2)
    while os.path.isdir(figures_dir) or os.path.isdir(logs_dir) or os.path.isdir(models_dir):
        # list existing files
        if os.path.isdir(figures_dir):
            figure_files = [os.path.join(figures_dir, f) for f in os.listdir(figures_dir) if
                            f.endswith('.jpg') or f.endswith('.png')]
        else:
            figure_files = []

        # check for .log files
        if os.path.isdir(logs_dir):
            log_files = [os.path.join(logs_dir, l) for l in os.listdir(logs_dir) \
                         if os.path.isfile(os.path.join(logs_dir, l))] \
                        + [os.path.join(target_exp_dir, f) for f in os.listdir(target_exp_dir) if f.endswith('.log')]
        else:
            log_files = []

        # check for model files
        if os.path.isdir(models_dir):
            model_files = [os.path.join(models_dir, m) for m in os.listdir(models_dir) \
                           if os.path.isfile(os.path.join(models_dir, m))]
        else:
            model_files = []

        if prompt_delete_existing and (len(figure_files) > 0 or len(log_files) > 0 or len(model_files) > 0):
            # TODO: print some of the latest figures, logs and models so we can see what epoch
            # these experiments trained until
            print(
                'Remove \n\t{} figures from {}\n\t{} logs from {}\n\t{} models from {}?[y]es / [n]o (create new folder) / [C]ontinue existing / remove [m]odels too: [y/n/C/m]'.format(
                    len(figure_files), figures_dir, len(log_files), logs_dir, len(model_files), models_dir))

            if debug_delete_input:
                print('Debug input: {}'.format(debug_delete_input))
                choice = debug_delete_input
            else:
                choice = input().lower()

            remove_choices = ['yes', 'y', 'ye']
            make_new_choices = ['no', 'n']
            continue_choices = ['c', '']
            remove_models_too = ['m']

            if choice in remove_choices:
                for f in figure_files + log_files:
                    print('Removing {}'.format(f))
                    os.remove(f)
            elif choice in remove_models_too:
                for f in figure_files + log_files + model_files:
                    print('Removing {}'.format(f))
                    os.remove(f)
            elif choice in continue_choices:
                print('Continuing in existing folder...')
                break

            elif choice in make_new_choices:
                copy_count += 1
                experiment_name = experiment_base_name + '_{}'.format(copy_count)
                target_exp_dir = os.path.join(exp_root, experiment_name)

                figures_dir = os.path.join(exp_root, experiment_name, 'figures')
                logs_dir = os.path.join(exp_root, experiment_name, 'logs')
                models_dir = os.path.join(exp_root, experiment_name, 'models')
        else:
            break

    if do_rename:
        # simply rename the existing old_exp_dir to exp_dir, rather than creating a new one
        os.rename(existing_exp_dir, target_exp_dir)
    else:
        # create each directory
        if not os.path.isdir(target_exp_dir):
            os.mkdir(target_exp_dir)

    # make subdirectories if they do not exist already
    if not os.path.isdir(figures_dir):
        os.mkdir(figures_dir)
    if not os.path.isdir(logs_dir):
        os.mkdir(logs_dir)
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    return experiment_name, target_exp_dir, figures_dir, logs_dir, models_dir


def _test_make_output_dirs():
    # setup for all unit tests
    exp_root = '_unit_test_experiments'
    if not os.path.isdir(exp_root):
        os.mkdir(exp_root)

    model_name = '_unit_test'

    # # unit test 0: make sure that we create all the subdirectories in a new experiment
    # print('Unit test 0 - Creating subdirectories')
    # new_model_name, exp_dir, figures_dir, logs_dir, models_dir = make_output_dirs(
    #     model_name, exp_root=exp_root)
    # assert new_model_name == model_name
    # assert os.path.isdir(figures_dir)
    # assert os.path.isdir(logs_dir)
    # assert os.path.isdir(models_dir)
    # os.rmdir(figures_dir)
    # os.rmdir(logs_dir)
    # os.rmdir(models_dir)
    # os.rmdir(exp_dir)
    # print('Unit test 0 - Creating subdirectories -- PASSED\n')
    #
    # # unit test 1: make sure that an experiment gets placed into the correct existing directory,
    # # if the original experiment was a copy
    # print('Unit test 1 - Putting experiment into _1 dir if it exists')
    # existing_exp_name = '_unit_test_1'
    # existing_exp_dir = os.path.join(exp_root, existing_exp_name)
    # if not os.path.isdir(existing_exp_dir):
    #     os.mkdir(existing_exp_dir)
    # new_model_name, exp_dir, figures_dir, logs_dir, models_dir = make_output_dirs(
    #     model_name, existing_exp_dir=existing_exp_dir, prompt_delete_existing=False,
    #     exp_root=exp_root)
    #
    # assert new_model_name == existing_exp_name
    # os.rmdir(figures_dir)
    # os.rmdir(logs_dir)
    # os.rmdir(models_dir)
    # os.rmdir(exp_dir)
    # print('Unit test 1 - Putting experiment into _1 dir if it exists -- PASSED\n')
    #
    # # unit test 2: make sure that we do not try to rename an experiment if
    # # the naming scheme has changed but we do not want to prompt rename
    # print('Unit test 2 - No renaming to updated name on prompt_update_name==False')
    # existing_exp_name = '_unit_test_2_oldscheme'
    # existing_exp_dir = os.path.join(exp_root, existing_exp_name)
    # if not os.path.isdir(existing_exp_dir):
    #     os.mkdir(existing_exp_dir)
    # new_model_name, exp_dir, figures_dir, logs_dir, models_dir = make_output_dirs(
    #     model_name, existing_exp_dir=existing_exp_dir,
    #     prompt_delete_existing=False, prompt_update_name=False,
    #     exp_root=exp_root)
    #
    # assert new_model_name == existing_exp_name
    # os.rmdir(figures_dir)
    # os.rmdir(logs_dir)
    # os.rmdir(models_dir)
    # os.rmdir(exp_dir)
    # print('Unit test 2 - No renaming to updated name on prompt_update_name==False -- PASSED\n')
    #
    # # unit test 3: make sure that we rename an existing dir to the new naming scheme
    # # if we answer yes to the prompt
    # print('Unit test 3 - Renamed to updated name on prompt_update_name==True and input==y')
    # existing_exp_dir = os.path.join(exp_root, '_unit_test_3_oldscheme')
    # if not os.path.isdir(existing_exp_dir):
    #     os.mkdir(existing_exp_dir)
    #
    # new_model_name, exp_dir, figures_dir, logs_dir, models_dir = make_output_dirs(
    #     model_name, existing_exp_dir=existing_exp_dir,
    #     prompt_delete_existing=False, prompt_update_name=True,
    #     debug_rename_input='y', debug_delete_input='n',
    #     exp_root=exp_root
    # )
    #
    # assert new_model_name == model_name
    # print('Looking for subdirs {}'.format(figures_dir))
    # assert os.path.normpath(figures_dir) == os.path.normpath(os.path.join(exp_root, model_name, 'figures'))
    # assert os.path.isdir(figures_dir)
    # assert os.path.isdir(logs_dir)
    # assert os.path.isdir(models_dir)
    # os.rmdir(figures_dir)
    # os.rmdir(logs_dir)
    # os.rmdir(models_dir)
    # os.rmdir(exp_dir)
    # print('Unit test 3 - Renamed to updated name on prompt_update_name==True and input==y -- PASSED\n')

    # unit test 4: make sure that we rename an existing dir to the new naming scheme
    # if we answer yes to the prompt
    print('Unit test 4 - Creating a copy when we answer n to delete existing dir')
    existing_exp_dir = os.path.join(exp_root, model_name)
    # create subdirectories first and put in a dummy file
    _, _, figures_dir, logs_dir, _ = make_output_dirs(model_name, exp_root=exp_root)
    existing_dummy_file = os.path.join(logs_dir, 'dummy.txt')
    with open(existing_dummy_file, 'w') as f:
        f.writelines('dummy file for unit test')


    new_model_name, exp_dir, figures_dir, logs_dir, models_dir = make_output_dirs(
        model_name, existing_exp_dir=existing_exp_dir,
        prompt_delete_existing=True, prompt_update_name=False,
        debug_delete_input='n',
        exp_root=exp_root
    )

    assert new_model_name == model_name + '_1'
    assert os.path.isdir(figures_dir)
    assert os.path.isdir(logs_dir)
    assert os.path.isdir(models_dir)
    # make sure we didnt remove the existing files
    assert os.path.isfile(existing_dummy_file)
    os.remove(existing_dummy_file)
    os.rmdir(figures_dir)
    os.rmdir(logs_dir)
    os.rmdir(models_dir)
    os.rmdir(exp_dir)
    print('Unit test 4 - Creating a copy when we answer n to delete existing dir -- PASSED\n')

    # unit test 5: make sure that we delete an existing dir if we say y to delete
    print('Unit test 5 - Replacing dir when we answer y to delete existing dir')
    existing_exp_dir = os.path.join(exp_root, '_unit_test')
    # create subdirectories first and put in a dummy file
    _, _, figures_dir, logs_dir, _ = make_output_dirs(model_name, exp_root=exp_root)
    existing_dummy_file = os.path.join(logs_dir, 'dummy.txt')
    with open(existing_dummy_file, 'w') as f:
        f.writelines('dummy file for unit test')

    new_model_name, exp_dir, figures_dir, logs_dir, models_dir = make_output_dirs(
        model_name, existing_exp_dir=existing_exp_dir,
        prompt_delete_existing=True, prompt_update_name=False,
        debug_delete_input='y',
        exp_root=exp_root
    )

    assert new_model_name == '_unit_test'
    # make sure we deleted the existing files
    assert not os.path.isfile(existing_dummy_file)
    assert os.path.isdir(figures_dir)
    assert os.path.isdir(logs_dir)
    assert os.path.isdir(models_dir)
    os.rmdir(figures_dir)
    os.rmdir(logs_dir)
    os.rmdir(models_dir)
    os.rmdir(exp_dir)
    print('Unit test 5 - Replacing dir when we answer y to delete existing dir -- PASSED\n')

    for d in os.listdir(exp_root):
        if os.path.isdir(os.path.join(exp_root, d)):
            os.rmdir(os.path.join(exp_root, d))
    os.rmdir(exp_root)

def get_latest_epoch_in_dir( d, match_prefixes = [] ):
    model_files = [ f for f in os.listdir(d) if f.endswith('.h5') ]

    epoch_nums = [ re.search( '(?<=epoch)[0-9]*', os.path.basename(f) ).group(0) for f in model_files ]
    epoch_nums = list(set([ int(n) for n in epoch_nums if n is not None  and n is not '']))
    max_epoch_num = 0

    if len(epoch_nums) == 0:
        return None

    if len(match_prefixes) > 0:
        for n in reversed(epoch_nums):
            curr_filenames = [os.path.basename(f) for f in model_files if 'epoch{}'.format(n) in f ]
            if np.all( [ np.any( [p in f for f in curr_filenames]) for p in match_prefixes] ) and n > max_epoch_num:
                max_epoch_num = n
    else:
#        elif len(match_prefixes) == 0 and n>max_epoch_nums
#            epoch_nums = [ re.search( '(?<=epoch)[0-9]*', os.path.basename(f) ).group(0) for f in model_files if ]

            max_epoch_num = max(epoch_nums)
    return max_epoch_num


def load_params_from_dir(in_dir):
    with open(os.path.join(in_dir, 'arch_params.json'), 'r') as f:
        arch_params = json.load(f)
    with open(os.path.join(in_dir, 'data_params.json'), 'r') as f:
        data_params = json.load(f)
    return arch_params, data_params

if __name__ == '__main__':
    _test_make_output_dirs()
