import os
import numpy as np
import glob
import datetime
figures_dir = './figures'
import argparse
import time

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('min_h_since_modified', type=float, default=24, nargs='?')
    ap.add_argument('-d', '--exp_dir', nargs='?', type=str, default='experiments', dest='experiments_root')
    ap.add_argument('-f', '--filter', nargs='*', default=[], type=str,
                    help='List of terms to filter for (if multiple, they will be ANDed)')
    ap.add_argument('-r', '--repeat_every', type=float, default=60, nargs='?')
    args = ap.parse_args()

    while True:
        all_experiments_figs_dirs = []

        # check all experiments in experiments_root for figures directories
        for exp_dir in os.listdir(args.experiments_root):
            if os.path.isdir(os.path.join(args.experiments_root, exp_dir)):
                for sub_dir in os.listdir(os.path.join(args.experiments_root, exp_dir)):
                    # look only for the figures subdirectory in each experiment
                    if 'figures' in sub_dir and os.path.isdir(os.path.join(args.experiments_root, exp_dir, sub_dir)):
                        all_experiments_figs_dirs.append(os.path.join(args.experiments_root, exp_dir, sub_dir))

        # get modified times for each figures directory
        dir_modified_times = [datetime.datetime.fromtimestamp(os.path.getmtime(figs_dir)) for figs_dir in all_experiments_figs_dirs]

        # sort dirs and dir modified times by modified time
        sorted_dirs, sorted_times = [list(x) for x in zip(*sorted(zip(all_experiments_figs_dirs, dir_modified_times),
                                                                  key=lambda dir_time_pair:dir_time_pair[1]))]

        time_since_modified = [(datetime.datetime.now() - t) for t in sorted_times]
        hours_since_modified = [t.days * 24 + t.seconds / 3600. for t in time_since_modified]

        n_experiment_dirs = len(hours_since_modified)

        # show at most the 10 most recently modified experiments
        print('Hours since modified\tDir')
        for i in range(max(n_experiment_dirs - 10, 0), n_experiment_dirs):
            print('{}\t\t\t{}'.format(round(hours_since_modified[i], 1), sorted_dirs[i]))

        latest_dirs = [d for i, d in enumerate(sorted_dirs) if hours_since_modified[i] < args.min_h_since_modified]
        if len(args.filter) > 0:
            latest_dirs = [d for d in latest_dirs if np.all([ft in d for ft in args.filter])]
        model_names = [os.path.basename(os.path.split(ld)[0]) for ld in latest_dirs]

        print('Combining images from each of {}'.format(model_names))
        os.system('python ~/evolving_wilds/scripts/combine_images.py {} -out_names {}'.format(' '.join(latest_dirs),
                                                                                              ' '.join(model_names)))
        time.sleep(args.repeat_every)
