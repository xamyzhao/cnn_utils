import json
import logging
import os
import sys

import file_utils
import numpy as np
from keras.utils import plot_model


class Experiment(object):
	def get_model_name(self):
		self.model_name = self.model_name.replace(' ', '') \
			.replace(')', '').replace('(', '') \
			.replace('[', '').replace(']', '') \
			.replace(',', '-').replace("'", '') \
			.replace('n_convs_per_stage', 'ncps').replace('enc_chans', 'enc').replace('use_upsample', 'upsamp')
		exp_name = self.model_name

		return exp_name

	def __init__(self, data_params, arch_params, exp_root='experiments', prompt_delete=False):
		self.do_profile = False

		self.arch_params = arch_params
		self.data_params = data_params

		# figure out model name0
		self.get_model_name()

		if 'exp_dir' in self.arch_params.keys():
			exp_dir = self.arch_params['exp_dir']
		else:
			exp_dir = None

		# make directories to store outputs
		self.model_name, \
		self.exp_dir, \
		self.figures_dir, self.logs_dir, self.models_dir \
			= file_utils.make_output_dirs(
			self.model_name,
			exp_root='./{}/'.format(exp_root), prompt_delete=prompt_delete, exp_dir=exp_dir)

		self.arch_params['exp_dir'] = self.exp_dir

		self.validation_losses_buffer = []
		self._init_logger()

	def get_dirs(self):
		return self.exp_dir, self.figures_dir, self.logs_dir, self.models_dir

	def load_data(self):
		with open(os.path.join(self.exp_dir, 'data_params.json'), 'w') as f:
			json.dump(self.data_params, f)

	def create_models(self):
		self._print_models()
		return None

	def load_models(self, load_epoch=None, stop_on_missing=True, init_layers=False):
		if load_epoch == 'latest':
			load_epoch = file_utils.get_latest_epoch_in_dir(self.models_dir)
			self.logger.debug('Found latest epoch {} in dir {}'.format(load_epoch, self.models_dir))

		if load_epoch is not None and int(load_epoch) > 0:
			self.logger.debug('Looking for models in {}'.format(self.models_dir))
			found_a_model = False
			model_files = os.listdir(self.models_dir)

			for m in self.models:
				# model_filename = os.path.join(models_dir, '{}_epoch{}.h5'.format(m.name, load_epoch))
				model_filename = [os.path.join(self.models_dir, mf) for mf in model_files \
				                  if mf.split('_epoch')[0] == m.name and 'epoch{}'.format(load_epoch) in mf]
				if len(model_filename) == 0:
					self.logger.debug('Could not find any model files with {}!'.format(m.name))
					model_filename = None
					if stop_on_missing:
						sys.exit()
					continue

				else:
					model_filename = model_filename[0]

				if os.path.isfile(model_filename):
					self.logger.debug('Loading model {} from {}'.format(m.name, model_filename))
					# m.summary()
					try:
						m.load_weights(model_filename)
					except ValueError:
						self.logger.debug('FAILED TO LOAD WEIGHTS DIRECTLY')
						if not init_layers:
							sys.exit()
					# self.logger.debug('FAILED loading weights, attempting to load model!')
					# from keras.models import load_model
					# m.summary(line_length=120)
					# m = load_model(model_filename)
					# m.summary(line_length=120)
					found_a_model = True
				elif not os.path.isfile(model_filename):
					self.logger.debug('Could not find model file {}!'.format(model_filename))
					if stop_on_missing:
						sys.exit()

			if not found_a_model:
				self.logger.debug('Did not find any models with epoch {} in dir {}!'.format(load_epoch, models_dir))
				load_epoch = 0

			self.latest_epoch = int(load_epoch) + 1
			return int(load_epoch) + 1
		else:
			return 0

	def create_generators(self, batch_size):
		print('create_generators not implemented')

	def compile_models(self):
		with open(os.path.join(self.exp_dir, 'arch_params.json'), 'w') as f:
			json.dump(self.arch_params, f)

	def _print_models(self, save_figs=True, figs_dir=None, models_to_print=None):
		if figs_dir is None:
			figs_dir = self.exp_dir

		# we might wnat to print some models but not save them all in self.models
		if models_to_print is None and hasattr(self, 'models_to_print'):
			models_to_print = self.models_to_print
		elif models_to_print is None:
			models_to_print = self.models

		for m in models_to_print:
			print(m.name)
			m.summary(line_length=120)

			if save_figs:
				plot_model(
					m,
					to_file=os.path.join(figs_dir, m.name + '.jpg'),
					show_shapes=True)
				with open(os.path.join(figs_dir, m.name + '.txt'), 'w') as fh:
					m.summary(print_fn=lambda x: fh.write(x + '\n'), line_length=120)

	def save_models(self, epoch, iter_count=None):
		for m in self.models:
			print('Saving {}'.format(m.name))
			if iter_count is not None:
				m.save(os.path.join(self.models_dir, '{}_epoch{}_iter{}.h5'.format(m.name, epoch, iter_count)))
			else:
				m.save(os.path.join(self.models_dir, '{}_epoch{}.h5'.format(m.name, epoch)))
		return 0

	def save_exp_info(self, exp_dir, figures_dir, models_dir, logs_dir):
		self.exp_dir = exp_dir
		self.figures_dir = figures_dir
		self.logs_dir = logs_dir
		self.models_dir = models_dir

	def _init_logger(self):
		if not hasattr(self, 'logger'):
			self.logger = None

		if self.logger is None:
			formatter = logging.Formatter('[%(asctime)s] %(message)s', "%Y-%m-%d %H:%M:%S")
			lfh = logging.FileHandler(filename=os.path.join(self.exp_dir, 'experiment.log'))
			lsh = logging.StreamHandler(sys.stdout)
			lfh.setFormatter(formatter)
			lsh.setFormatter(formatter)
			lfh.setLevel(logging.DEBUG)
			lsh.setLevel(logging.DEBUG)

			self.logger = logging.getLogger(self.__class__.__name__)
			self.logger.setLevel(logging.DEBUG)
			self.logger.handlers = []
			self.logger.addHandler(lfh)
			self.logger.addHandler(lsh)

		if self.do_profile:
			formatter = logging.Formatter('[%(asctime)s] %(message)s', "%Y-%m-%d %H:%M:%S")
			lfh = logging.FileHandler(filename=os.path.join(self.exp_dir, 'profiler.log'))
			lfh.setFormatter(formatter)
			lfh.setLevel(logging.DEBUG)

			self.profiler_logger = logging.getLogger(self.__class__.__name__ + '_profiler')
			self.profiler_logger.handlers = []
			self.profiler_logger.setLevel(logging.DEBUG)
			self.profiler_logger.addHandler(lfh)

	def _reopen_log_file(self):
		if self.logger is not None:
			self.logger.handlers[0].close()
			lfh = logging.FileHandler(filename=os.path.join(self.exp_dir, 'experiment.log'))
			self.logger.handlers[0] = lfh

		if self.profiler_logger is not None:
			self.profiler_logger.handlers[0].close()
			lfh = logging.FileHandler(filename=os.path.join(self.exp_dir, 'profiler.log'))
			self.profiler_logger.handlers[0] = lfh

	def make_results_im(self):
		return np.zeros((8, 8, 3))

	def get_n_train(self):
		return self.X_source_train.shape[0]

	def train_discriminator(self):
		print('train_discriminator not implemented')

	def train_joint(self):
		print('train_joint not implemented')

	def update_epoch_count(self, epoch):
		return 0