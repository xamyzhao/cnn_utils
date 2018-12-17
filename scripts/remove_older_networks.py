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

def remove_all_but_milestones_and_recent( in_dir, file_exts, num_prefix, milestone_interval, num_recent_to_keep = 3, test=False ):
	files = [ os.path.join( in_dir, f) for f in os.listdir( in_dir ) if np.any( [f.endswith(e) for e in file_exts])  \
									if re.search('(?<={})[0-9]*'.format( num_prefix),f) is not None ]

	file_times = [ datetime.datetime.fromtimestamp( os.path.getmtime(f)) for f in files ]
	nums = [ int(re.search('(?<={})[0-9]*'.format( num_prefix ), os.path.basename(f) ).group(0)) for f in files ]

#	print('[{}] {}s in {}: {}'.format(  datetime.datetime.now(), num_prefix, in_dir, sorted(list(set(nums)))))
	if(len(nums)> num_recent_to_keep ):
#				model_nums = [int(x) for x in model_nums]
		#model_nums.sort()
#		nums, files, file_times = [list(sl) for sl in zip(*sorted(zip(nums, files, file_times ), key=lambda tup:tup[2], reverse=True))]
#		print(file_times)
#		print(nums)
#		files_to_remove = files[:-num_recent_to_keep] 
		nums_to_remove = sorted(list(set(nums)))[:-num_recent_to_keep]
		
		if len(nums_to_remove) == 0:
			return
		nums_to_remove = keep_closest_to_milestones(milestone_interval, nums_to_remove)
		if len(nums_to_remove) == 0:
			return
	

		files_to_remove = [f for f in files if np.any( ['{}{}_'.format(num_prefix, ntr) in os.path.basename(f) or '{}{}.'.format(num_prefix,ntr) in os.path.basename(f) for ntr in nums_to_remove]) ]
		 
#		print(nums_to_remove)
#		for i,ntr in enumerate( nums_to_remove ):
#			if ntr % milestone_interval > 0:
#				files_to_remove = [ f for f in files if '{}{}'.format(num_prefix, ntr) in f ]
		if test:
			print('[{}] Will remove:'.format(datetime.datetime.now().strftime('%m-%d-%y %H:%M')))
			for f in files_to_remove:
				print(f)
		else:		
			print('[{}] Removed:'.format(datetime.datetime.now().strftime('%m-%d-%y %H:%M')))
			for f in files_to_remove:
				print(f)
				os.remove(f)
#			print(files_to_remove )
#							os.remove( fileToRemove )

exp_dir = './experiments'
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
	ap.add_argument( '-nd', nargs='?', type=str, help='networks dir', default = './models' )
	ap.add_argument( '-od', nargs='?', type=str, help='figures dir', default = './figures' )
	ap.add_argument( '-models_milestone', nargs='?', type=int, help='save any files with epochs % milestone == 0', default = 100 )
	ap.add_argument( '-figs_milestone', nargs='?', type=int, help='save any files with epochs % milestone == 0', default = 10 )
	ap.add_argument( '-model_num_prefix', nargs='?', type=str, help='number prefix e.g. epoch, batch, iter', default='epoch' )
	ap.add_argument( '-fig_num_prefix', nargs='?', type=str, help='number prefix e.g. epoch, batch, iter', default='epoch' )
	ap.add_argument( '-test', action='store_true', help='turn on to see what will be removed without actually removing', default = False )
	ap.add_argument( '-m', action='store_true', help='Remove from models dir', default = False )
	ap.add_argument( '-f', action='store_true', help='Remove from figures dir', default = False )
	args = ap.parse_args()

	nd = args.nd
	od = args.od

	
	model_ext = ['.h5']
	fig_ext = ['.mat','.jpg','.png']
	while True:
		if args.m:
			for model_folder in [ os.path.join(exp_dir, ed, nd ) for ed in os.listdir(exp_dir) \
							if os.path.isdir( os.path.join(exp_dir,ed,nd) ) ]:
				remove_all_but_milestones_and_recent( in_dir = model_folder, file_exts = model_ext, 
																							num_prefix = args.model_num_prefix, 
																							milestone_interval = args.models_milestone,
																							num_recent_to_keep = num_models_to_keep,
																							test = args.test )
		if args.f: 
			for model_folder in [ os.path.join( exp_dir, ed, od) for ed in os.listdir(exp_dir) \
							if os.path.isdir(os.path.join(exp_dir,ed,od))]:
				remove_all_but_milestones_and_recent( in_dir = model_folder, file_exts = fig_ext, 
																							num_prefix = args.fig_num_prefix, 
																							milestone_interval = args.figs_milestone,
																							num_recent_to_keep = num_figs_to_keep,
																							test = args.test )
				
		'''
			fig_files = [ os.path.join( model_folder, f) for f in os.listdir( model_folder ) if f.endswith('.h5') \
											if re.search('(?<={})[0-9]*(?=.h5)'.format(args.fig_num_prefix),f) is not None ]

			fig_nums = [ int(re.search('(?<={})[0-9]*(?=.h5)'.format(args.fig_num_prefix), os.path.basename(f) ).group(0)) for f in fig_files ]
	 
			if(len(fig_nums)> num_figs_to_keep ):
				idxs_to_remove = []
				for i in range(0,len(fig_nums)-num_figs_to_keep+1):
					if fig_nums[i] % args.figs_milestone > 0 and fig_nums[i] > 2*args.figs_milestone:
						idxs_to_remove.append(i)
				
	#			model_nums = [int(x) for x in model_nums]
	#			model_nums.sort()
				print('{} Files in {}: {}'.format(  datetime.datetime.now(), model_folder, fig_nums))

	#			model_numsToRemove = model_nums[:-num_models_to_keep] 
				for i in idxs_to_remove:
#					fileToRemove = od + modelFolder + '/' + str(mftr)
					fileToRemove = fig_files[i]
					if args.test:
						print( 'Will remove {}'.format(fileToRemove))
					else:
						print('{} Removed: {}'.format(  datetime.datetime.now(), fileToRemove ))
		#				os.remove( fileToRemove )
		'''
		print('Sleeping for 600s...')
		time.sleep( 600 )


