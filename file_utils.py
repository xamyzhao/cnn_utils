import json
import os
import re
#def filenames_to_im_ids( fns ):
#	return [ filename_to_id(fn) for fn in fns ]

#def filename_to_id( fn ):
#	return fn.split('_')[0]
import numpy as np


def filenames_to_im_ids( im_files ):
	if isinstance(im_files[0], int):
		return im_files
	elif 'frame_' in im_files[0]:
		im_file_ids = [int(re.search('(?<=frame_)[0-9]*', f).group(0)) for f in im_files]
		return im_file_ids
	elif 'frame' in im_files[0]:
		im_file_ids = [int(re.search('(?<=frame)[0-9]*', f).group(0)) for f in im_files]
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
	#		im_file_ids = [ ''.join([str(ord(c)) for c in ]) for f in im_files ]
	#		print(im_file_ids)
	return im_file_ids


def make_output_dirs(base_model_name, prompt_delete=True, exp_root ='./experiments/', exp_dir=None):
	fig_root = './figures/'
	log_root = './logs/'
	model_root = './models/'

	if exp_dir is None:
		exp_dir = exp_root + base_model_name
		model_name = base_model_name
	else:
		model_name = os.path.basename(exp_dir)

	figures_dir = os.path.join(exp_dir, 'figures')
	logs_dir = os.path.join(exp_dir, 'logs')
	models_dir = os.path.join(exp_dir, 'models')

	copy_count = 0

	while os.path.isdir(figures_dir) or os.path.isdir(logs_dir) or os.path.isdir(models_dir):
		# list existing files
		if os.path.isdir(figures_dir):
			figure_files = [os.path.join(figures_dir, f) for f in os.listdir(figures_dir) if
							f.endswith('.jpg') or f.endswith('.png')]
		else:
			figure_files = []

		if os.path.isdir(logs_dir):
			log_files = [os.path.join(logs_dir, l) for l in os.listdir(logs_dir) \
						 if os.path.isfile(os.path.join(logs_dir, l))] \
						+ [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith('.log')]
			# also includde any .log files
		else:
			log_files = []

		if os.path.isdir(models_dir):
			model_files = [os.path.join(models_dir, m) for m in os.listdir(models_dir) \
						   if os.path.isfile(os.path.join(models_dir, m))]
		else:
			model_files = []

		if prompt_delete and (len(figure_files) > 0 or len(log_files) > 0 or len(model_files) > 0):
			print(
				'Remove \n\t{} figures from {}\n\t{} logs from {}\n\t{} models from {}?\n[y]es / [n]o (create new folder) / [C]ontinue existing / remove [m]odels too: [y/n/C/m]'.format(
					len(figure_files), figures_dir, len(log_files), logs_dir, len(model_files), models_dir))

			try:
				choice = raw_input().lower()
			except NameError:
				# python 3 syntax
				choice = input().lower()

			#			if len(choice) == 0:
			#				choice=['yes']
			remove_choices = ['yes', 'y', 'ye']
			make_new_choices = ['no', 'n']
			continue_choices = ['c', '']
			remove_models_too = ['m']

			#			for c in choice:
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
				model_name = base_model_name + '_{}'.format(copy_count)
				exp_dir = exp_root + model_name

				figures_dir = exp_root + model_name + '/figures'
				logs_dir = exp_root + model_name + '/logs'
				models_dir = exp_root + model_name + '/models'
		else:
			break

	if not os.path.isdir(exp_dir):
		os.mkdir(exp_dir)
	if not os.path.isdir(figures_dir):
		os.mkdir(figures_dir)
	if not os.path.isdir(logs_dir):
		os.mkdir(logs_dir)
	if not os.path.isdir(models_dir):
		os.mkdir(models_dir)
	return model_name, exp_dir, figures_dir, logs_dir, models_dir

def _test_make_output_dirs():
	model_name = '_test_make_output_dirs'
	new_model_name, figures_dir, logs_dir, models_dir = make_output_dirs(model_name)
	assert new_model_name == model_name
	assert os.path.isdir( figures_dir )
	assert os.path.isdir( logs_dir )
	assert os.path.isdir( models_dir )


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
#		elif len(match_prefixes) == 0 and n>max_epoch_nums
#			epoch_nums = [ re.search( '(?<=epoch)[0-9]*', os.path.basename(f) ).group(0) for f in model_files if ]

			max_epoch_num = max(epoch_nums)
	return max_epoch_num


def load_params_from_dir(in_dir):
	with open(os.path.join(in_dir, 'arch_params.json'), 'r') as f:
		arch_params = json.load(f)
	with open(os.path.join(in_dir, 'data_params.json'), 'r') as f:
		data_params = json.load(f)
	return arch_params, data_params
