import os
import re
import numpy as np

import argparse


import time
import datetime

num_models_to_keep = 8
num_figs_to_keep = 8


def keep_closest_to_milestones(milestone_interval, nums_to_remove):
    milestone_nums = np.arange(0, max(nums_to_remove)+1, milestone_interval).astype(int)

    nums_to_keep = []
    # keep the closest example to each milestone
    for mn in milestone_nums:
        dists_from_milestone = np.asarray(nums_to_remove) - mn
        nums_to_keep.append(nums_to_remove[np.argmin(np.abs(dists_from_milestone))])
    print('Keeping {}'.format(nums_to_keep))
    nums_to_remove = [n for n in nums_to_remove if n not in nums_to_keep]
    print('Removing {}'.format(nums_to_remove))
    return nums_to_remove


def remove_all_but_milestones_and_recent(
        in_dir,
        file_exts,
        num_prefix,
        milestone_interval, num_recent_to_keep=3,
        debug=False):
    files_to_remove = [
        os.path.join(in_dir, f) for f in os.listdir(in_dir)
        if np.any([f.endswith(e) for e in file_exts]) \
        if re.search('(?<={})[0-9]*'.format(num_prefix), f) is not None]

    file_times = [datetime.datetime.fromtimestamp(os.path.getmtime(f)) for f in files_to_remove]
    nums = [int(re.search('(?<={})[0-9]*'.format(num_prefix), os.path.basename(f)).group(0)) for f in files_to_remove]

    if (len(nums) > num_recent_to_keep):
        nums_to_remove = sorted(list(set(nums)))[:-num_recent_to_keep]

        if len(nums_to_remove) == 0:
            return
        nums_to_remove = keep_closest_to_milestones(milestone_interval, nums_to_remove)
        if len(nums_to_remove) == 0:
            return

        files_to_remove = [f for f in files_to_remove if np.any(['{}{}_'.format(num_prefix, ntr) in os.path.basename(
            f) or '{}{}.'.format(num_prefix, ntr) in os.path.basename(f) for ntr in nums_to_remove])]

        if debug:
            print('[{}] Will remove:'.format(datetime.datetime.now().strftime('%m-%d-%y %H:%M')))
            for f in files_to_remove:
                print(f)
        else:
            print('[{}] Removed:'.format(datetime.datetime.now().strftime('%m-%d-%y %H:%M')))
            for f in files_to_remove:
                print(f)
                os.remove(f)


exp_root = './experiments'
if __name__ == '__main__':
    import sys

    # unit test
    remove_nums = keep_closest_to_milestones(100, np.arange(0, 200, 21).astype(int))
    assert 0 not in remove_nums
    assert 105 not in remove_nums

    remove_nums = keep_closest_to_milestones(100, np.arange(0, 300, 100).astype(int))
    assert 100 not in remove_nums
    assert 200 not in remove_nums

    ap = argparse.ArgumentParser()
    ap.add_argument('-nd', nargs='?', type=str, help='networks dir', default='./models')
    ap.add_argument('-od', nargs='?', type=str, help='figures dir', default='./figures')
    ap.add_argument('-mm', '--models_milestone', nargs='?', type=int,
                    help='save any files with epochs % milestone == 0',
                    default=100)
    ap.add_argument('-fm', '--figs_milestone', nargs='?', type=int,
                    help='save any files with epochs % milestone == 0',
                    default=10)
    ap.add_argument('-model_num_prefix', nargs='?', type=str, help='number prefix e.g. epoch, batch, iter',
                    default='epoch')
    ap.add_argument('-fig_num_prefix', nargs='?', type=str, help='number prefix e.g. epoch, batch, iter',
                    default='epoch')
    ap.add_argument('--debug', action='store_true', help='turn on to see what will be removed without actually removing',
                    default=False)
    ap.add_argument('-m', action='store_true', help='Boolean to do removal from models dir', default=False,
                    dest='do_remove_models')
    ap.add_argument('-f', action='store_true', help='Boolean to do removal from figures dir', default=False,
                    dest='do_remove_figures')
    ap.add_argument('-r', action='store_true', help='Recursive, look in child dirs for models and figures dirs too',
                    dest='do_recursive',
                    default=False)
    args = ap.parse_args()

    model_dirname = args.nd
    figs_dirname = args.od

    model_extensions = ['.h5']
    figure_extensions = ['.mat', '.jpg', '.png']
    while True:
        model_dirs = [os.path.join(exp_root, ed, model_dirname) for ed in os.listdir(exp_root) \
                      if os.path.isdir(os.path.join(exp_root, ed, model_dirname))]
        if args.do_recursive:
            # recursively look for model dirs
            for ed in os.listdir(exp_root):
                # TODO: this only looks one level down
                for cd in os.listdir(os.path.join(exp_root, ed)):
                    models_dir = os.path.join(exp_root, ed, cd, model_dirname)
                    if os.path.isdir(models_dir):
                        model_dirs.append(models_dir)
        if args.do_remove_models:
            for model_folder in model_dirs:
                remove_all_but_milestones_and_recent(in_dir=model_folder, file_exts=model_extensions,
                                                     num_prefix=args.model_num_prefix,
                                                     milestone_interval=args.models_milestone,
                                                     num_recent_to_keep=num_models_to_keep,
                                                     debug=False)
        if args.do_remove_figures:
            fig_dirs = [os.path.join(exp_root, ed, figs_dirname) for ed in os.listdir(exp_root) \
                                if os.path.isdir(os.path.join(exp_root, ed, figs_dirname))]
            if args.do_recursive:
                # recursively look for model dirs
                for ed in os.listdir(exp_root):
                    # TODO: this only looks one level down
                    for cd in os.listdir(os.path.join(exp_root, ed)):
                        figs_dir = os.path.join(exp_root, ed, cd, figs_dirname)
                        if os.path.isdir(figs_dir):
                            fig_dirs.append(figs_dir)

            for figs_folder in fig_dirs:
                remove_all_but_milestones_and_recent(in_dir=figs_folder, file_exts=figure_extensions,
                                                     num_prefix=args.fig_num_prefix,
                                                     milestone_interval=args.figs_milestone,
                                                     num_recent_to_keep=num_figs_to_keep,
                                                     debug=False)

        print('Sleeping for 600s...')
        time.sleep(600)
