import os
import glob
import datetime
exps_dir = './experiments'
figures_dir = './figures'
import argparse
import time
import numpy as np

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-model_names', nargs='*', default=[], type=str, help='Model names to associate with latest dirs')
	ap.add_argument('-filter', nargs='*', default=[], type=str, help='List of terms to filter (AND) for')
	ap.add_argument('-port', nargs='?',type=int, help='port number', default=None)
	ap.add_argument('min_h',type=float,default=24, nargs='?')
	ap.add_argument('-repeat_every',type=float,default=60, nargs='?')
	args = ap.parse_args()
	print args.min_h

#	while True:
#		log_dirs = [ os.path.join(figures_dir,d) for d in os.listdir(figures_dir) if os.path.isdir( os.path.join( figures_dir, d) )]
	log_dirs = []
	for exp_dir in os.listdir(exps_dir):
		if os.path.isdir( os.path.join(exps_dir,exp_dir) ):
			for sub_dir in os.listdir(os.path.join(exps_dir,exp_dir)):
				if 'logs' in sub_dir and os.path.isdir( os.path.join(exps_dir,exp_dir,sub_dir) ):
					log_dirs.append( os.path.join(exps_dir,exp_dir,sub_dir))
	dir_times = [ datetime.datetime.fromtimestamp(os.path.getmtime(d)) for d in log_dirs] 

	sorted_dir_time_pairs = [ list(x) for x in zip(*sorted(zip(log_dirs, dir_times), key=lambda pair:pair[1])) ]
	sorted_dirs = sorted_dir_time_pairs[0]
	sorted_times = sorted_dir_time_pairs[1]
	time_since = [ (datetime.datetime.now() - t) for t in sorted_times ]
	hours_since = [ t.days*24 + t.seconds/3600. for t in time_since ]

	print('Hours since modified\tDir')
	for i in range( max(len(hours_since)-10,0),len(hours_since)):
		print('{}\t\t\t{}'.format( round(hours_since[i],1), sorted_dirs[i]))

	latest_dirs = [d[2:] for i,d in enumerate(sorted_dirs) if hours_since[i] < args.min_h ] #remove leading ./

	if len(args.filter) > 0:
		latest_dirs = [ d for d in latest_dirs if np.all( [ft in d for ft in args.filter] )]
	if len(args.model_names) == 0:
		model_names = [ os.path.basename( os.path.split( ld )[0] ) for ld in latest_dirs ]
	else:
		model_names = args.model_names
	tb_names = [ '{}:{}'.format(model_names[i], latest_dirs[i]) for i in range(len(latest_dirs)) ]
	tb_str = '{}'.format(','.join(tb_names))
	print(tb_str) 
	if args.port is not None:
		os.system('tensorboard --logdir={} --port={}'.format(tb_str, args.port))
	else:
		os.system('tensorboard --logdir={}'.format(tb_str))
#		time.sleep(args.repeat_every)
			
