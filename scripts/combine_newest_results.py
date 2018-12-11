import os
import numpy as np
import glob
import datetime
figures_dir = './figures'
import argparse
import time

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-exp_dir', nargs='?', type=str, default='experiments', dest='experiments_root')
	ap.add_argument('min_h',type=float,default=24, nargs='?')
	ap.add_argument('-filter', nargs='*', default=[], type=str, help='List of terms to filter (AND) for')
	ap.add_argument('repeat_every',type=float,default=60, nargs='?')
	args = ap.parse_args()
	print(args.min_h)

	while True:
#		fig_dirs = [ os.path.join(figures_dir,d) for d in os.listdir(figures_dir) if os.path.isdir( os.path.join( figures_dir, d) )]
		fig_dirs = []
		for exp_dir in os.listdir(args.experiments_root):
			if os.path.isdir( os.path.join(args.experiments_root, exp_dir) ):
				for sub_dir in os.listdir(os.path.join(args.experiments_root,exp_dir)):
					if 'figures' in sub_dir and os.path.isdir( os.path.join(args.experiments_root,exp_dir,sub_dir) ):
						fig_dirs.append( os.path.join(args.experiments_root,exp_dir,sub_dir))
#		fig_dirs = [ os.path.join( args.experiments_root, md, fd ) if os.path.isdir(os.path.join(args.experiments_root,md)) for md in os.listdir(args.experiments_root) for fd in os.listdir( os.path.join( args.experiments_root, md)) 
#			 and os.path.isdir( os.path.join( args.experiments_root, md,fd)) and 'figures' in fd ]
		dir_times = [ datetime.datetime.fromtimestamp(os.path.getmtime(d)) for d in fig_dirs] 

		sorted_dir_time_pairs = [ list(x) for x in zip(*sorted(zip(fig_dirs, dir_times), key=lambda pair:pair[1])) ]
		sorted_dirs = sorted_dir_time_pairs[0]
		sorted_times = sorted_dir_time_pairs[1]
		time_since = [ (datetime.datetime.now() - t) for t in sorted_times ]
		hours_since = [ t.days*24 + t.seconds/3600. for t in time_since ]

		print('Hours since modified\tDir')
		for i in range( max(len(hours_since)-10,0),len(hours_since)):
			print('{}\t\t\t{}'.format( round(hours_since[i],1), sorted_dirs[i]))


		latest_dirs = [d for i,d in enumerate(sorted_dirs) if hours_since[i] < args.min_h ]
		if len(args.filter) > 0:
			latest_dirs = [ d for d in latest_dirs if np.all( [ft in d for ft in args.filter] )]
		model_names = [ os.path.basename( os.path.split( ld )[0] ) for ld in latest_dirs ]
		print(model_names) 
		os.system('python ~/evolving_wilds/scripts/combine_images.py {} -out_names {}'.format( ' '.join(latest_dirs), ' '.join(model_names)))
		time.sleep(args.repeat_every)
				
