from basic_network_utils import cvae_modules

import sys
sys.path.append('../evolving_wilds')
from cnn_utils import metrics, basic_networks

class CVAE(object):
    def __init__(self,
                 ae_input_shapes,
                 ae_input_names,
                 conditioning_input_shapes,
                 conditioning_input_names,
                 output_shape,
                 source_im_idx=None,
                 mask_im_idx=None,
                 transform_latent_dim=50,
                 condition_on_image=True,
                 n_concat_scales=3,
                 transform_type='flow',
                 transform_activation=None,
                 flow_indexing='xy', color_transform_type=None,
                 n_outputs=1,
                 transform_enc_params=None,
                 dec_params=None,
                 include_aug_matrix=False,
                 aug_pad_vals=(1, 1),
                 clip_output_range=None,
        ):
        self.conditioning_input_shapes = [tuple(s) for s in conditioning_input_shapes]
        self.conditioning_input_names = conditioning_input_names

        self.ae_input_shapes = ae_input_shapes
        self.ae_input_names = ae_input_names

        self.output_shape = output_shape
        self.n_outputs = n_outputs # in case we need to return copies of the transformed image for multiple losses

        self.source_im_idx = source_im_idx
        self.mask_im_idx = mask_im_idx

        # correct for outdated spec of nf_enc
        self.transform_enc_params = transform_enc_params.copy()
        if 'nf_enc' not in self.transform_enc_params.keys():
            self.transform_enc_params['nf_enc'] = transform_enc_params['enc_chans']
        if 'nf_dec' not in self.transform_enc_params.keys():
            self.transform_enc_params['nf_dec'] = list(reversed(self.transform_enc_params['nf_enc']))
        if 'ks' not in self.transform_enc_params:
            self.transform_enc_params['ks'] = 3

        if dec_params is None:
            self.dec_params = self.transform_enc_params
        else:
            self.dec_params = dec_params

        uncommon_keys = ['fully_conv', 'use_residuals', 'use_upsample']
        for k in uncommon_keys:
            if k not in self.transform_enc_params.keys():
                self.transform_enc_params[k] = False

        self.condition_on_image = condition_on_image
        self.n_concat_scales = n_concat_scales

        self.include_aug_matrix = include_aug_matrix  # only used in train wrapper
        self.aug_pad_vals = aug_pad_vals

        self.transform_type = transform_type
        if self.transform_type is not None:
            self.transform_name = transform_type
        else:
            self.transform_name = 'synth'

        self.flow_indexing = flow_indexing
        self.color_transform_type = color_transform_type

        self.transform_latent_shape = (transform_latent_dim,)

        self.transform_activation = transform_activation
        self.clip_output_range = clip_output_range

    def create_modules(self):
        print('Creating CVAE with encoder params {}'.format(self.transform_enc_params))
        self.transform_enc_model = \
            cvae_modules.transform_encoder_model(
                input_shapes=self.ae_input_shapes,
                input_names=self.ae_input_names,
                latent_shape=self.transform_latent_shape,
                model_name='{}_transform_encoder_cvae'.format(self.transform_name),
                enc_params=self.transform_enc_params)

        self.transformer_model = \
            cvae_modules.transformer_concat_model(
                conditioning_input_shapes=self.conditioning_input_shapes,
                conditioning_input_names=self.conditioning_input_names,
                output_shape=self.output_shape,
                source_input_idx=self.source_im_idx,
                mask_by_conditioning_input_idx=self.mask_im_idx,
                model_name=self.transform_name + '_transformer_cvae',
                transform_type=self.transform_type,
                color_transform_type=self.color_transform_type,
                condition_on_image=self.condition_on_image,
                n_concat_scales=self.n_concat_scales,
                transform_latent_shape=self.transform_latent_shape,
                enc_params=self.dec_params,
                transform_activation=self.transform_activation,
                clip_output_range=self.clip_output_range
        )

    def create_train_wrapper(self, n_samples=1):
        self.trainer_model = \
            cvae_modules.cvae_trainer_wrapper(
                ae_input_shapes=self.ae_input_shapes,
                ae_input_names=self.ae_input_names,
                conditioning_input_shapes=self.conditioning_input_shapes,
                conditioning_input_names=self.conditioning_input_names,
                output_shape=self.output_shape,
                model_name='{}_transformer_cvae_trainer'.format(self.transform_name),
                transform_encoder_model=self.transform_enc_model,
                transformer_model=self.transformer_model,
                transform_type=self.transform_type,
                transform_latent_shape=self.transform_latent_shape,
                include_aug_matrix=self.include_aug_matrix,
                n_outputs=self.n_outputs
            )

        # TODO: include the conditional encoder if we have an AVAE
        self.tester_model = cvae_modules.cvae_tester_wrapper(
            conditioning_input_shapes=self.conditioning_input_shapes,
            conditioning_input_names=self.conditioning_input_names,
            dec_model=self.transformer_model,
            latent_shape=self.transform_latent_shape,
        )
        '''
        else:
            self.trainer_model = \
                VTE_network_modules.VTE_transformer_train_model(
                    img_shape=self.img_shape,
                    output_shape=self.output_shape,
                    model_name='{}_transformer_cvae_trainer'.format(self.transform_type),
                    vte_transform_encoder_model=self.transform_enc_model,
                    vte_transformer_test_model=self.transformer_model,
                    learn_transform_mask=self.learn_transform_mask,
                    apply_flow_transform=self.transform_type == 'flow',
                    apply_color_transform=self.transform_type == 'color',
                    include_transform_loss=True,
                    color_transform_type=self.color_transform_type,
                    transform_latent_shape=self.transform_latent_shape)
            self.tester_model = self.transformer_model
        '''
        self.vae_metrics = metrics.VAE_metrics(
            var_target=1.,
            mu_target=0.,
            axis=-1)

    def get_models(self):
        return [self.transform_enc_model, self.transformer_model]
#                self.trainer_model, self.tester_model]

    def _get_kl_losses(self):
        # KL losses
        loss_names = ['{}_kl_mu'.format(self.transform_type), '{}_kl_logsigma'.format(self.transform_type)]
        loss_fns = [self.vae_metrics.kl_mu, self.vae_metrics.kl_log_sigma]
        loss_weights = [1.] * 2
        return loss_names, loss_fns, loss_weights

    def get_losses(self,
                   transform_reg_fn=None, transform_reg_lambda=1., transform_reg_name='lapl',
                   recon_loss_fn=None, recon_loss_weight=1., recon_loss_name='l2'):

        loss_names = ['total']
        loss_fns = []
        loss_weights = []

        # convert to list so we can consistently process multiple losses
        if not isinstance(recon_loss_fn, list):
            recon_loss_fn = [recon_loss_fn]
            recon_loss_name = [recon_loss_name]
        if not isinstance(recon_loss_weight, list):
            recon_loss_weight = [recon_loss_weight]

        if self.transform_type is not None:
            # reconstruction first since this is what we care about. then smoothness
            loss_names += [
                '{}_recon_{}'.format(self.transform_name, rln) for rln in recon_loss_name
            ] + [
                '{}_smooth_{}'.format(self.transform_name, transform_reg_name),
            ]
            loss_fns += recon_loss_fn + [transform_reg_fn]  # smoothness reg, reconstruction
            loss_weights += recon_loss_weight + [transform_reg_lambda]
        else:  # no transform, just direct synthesis
            # smoothness and reconstruction loss first
            loss_names += [
                '{}_recon_{}'.format(self.transform_name, rln) for rln in recon_loss_name
            ]
            loss_fns += recon_loss_fn  # smoothness reg, reconstruction
            loss_weights += recon_loss_weight

        # KL mean and logvar losses at end
        loss_names_kl, loss_fns_kl, loss_weights_kl = self._get_kl_losses()
        loss_names += loss_names_kl
        loss_fns += loss_fns_kl
        loss_weights += loss_weights_kl
        return loss_names, loss_fns, loss_weights


    def get_train_targets(self, I, J, batch_size):
        zeros_latent = np.zeros((batch_size, ) + self.transform_latent_shape)

        train_targets = []

        if self.transform_type is not None:
            # smoothness reg
            train_targets.append(np.zeros((J.shape[:-1] + (2,))))
        # output image reconstruction
        train_targets.append(J)
        train_targets += [zeros_latent] * 2

        return train_targets



class CVAE_segment_selector(object):
    def __init__(self,
                 ae_input_shapes,
                 ae_input_names,
                 conditioning_input_shapes,
                 conditioning_input_names,
                 output_shape,
                 latent_distribution='normal',
                 source_im_idx=None,
                 mask_im_idx=None,
                 transform_latent_dim=50,
                 condition_on_image=True,
                 n_concat_scales=3,
                 transform_type='flow',
                 transform_activation=None,
                 flow_indexing='xy', color_transform_type=None,
                 n_outputs=1,
                 transform_enc_params=None,
                 dec_params=None,
                 include_aug_matrix=False,
                 aug_pad_vals=(1, 1),
                 clip_output_range=None,
                 ):
        self.conditioning_input_shapes = [tuple(s) for s in conditioning_input_shapes]
        self.conditioning_input_names = conditioning_input_names

        self.ae_input_shapes = ae_input_shapes
        self.ae_input_names = ae_input_names

        self.output_shape = output_shape
        self.n_outputs = n_outputs  # in case we need to return copies of the transformed image for multiple losses

        self.source_im_idx = source_im_idx
        self.mask_im_idx = mask_im_idx

        # standard normal, or categorical
        self.latent_distribution = latent_distribution

        # correct for outdated spec of nf_enc
        self.transform_enc_params = transform_enc_params.copy()
        if 'nf_enc' not in self.transform_enc_params.keys():
            self.transform_enc_params['nf_enc'] = transform_enc_params['enc_chans']
        if 'nf_dec' not in self.transform_enc_params.keys():
            self.transform_enc_params['nf_dec'] = list(reversed(self.transform_enc_params['nf_enc']))
        if 'ks' not in self.transform_enc_params:
            self.transform_enc_params['ks'] = 3

        if dec_params is None:
            self.dec_params = self.transform_enc_params
        else:
            self.dec_params = dec_params

        uncommon_keys = ['fully_conv', 'use_residuals', 'use_upsample']
        for k in uncommon_keys:
            if k not in self.transform_enc_params.keys():
                self.transform_enc_params[k] = False

        self.condition_on_image = condition_on_image
        self.n_concat_scales = n_concat_scales

        self.include_aug_matrix = include_aug_matrix  # only used in train wrapper
        self.aug_pad_vals = aug_pad_vals

        self.transform_type = transform_type
        if self.transform_type is not None:
            self.transform_name = transform_type
        else:
            self.transform_name = 'synth'

        self.flow_indexing = flow_indexing
        self.color_transform_type = color_transform_type

        self.transform_latent_shape = (transform_latent_dim,)

        self.transform_activation = transform_activation
        self.clip_output_range = clip_output_range

        # TODO: rename this class to be a segment selector
        if self.latent_distribution == 'categorical':
            self.n_segs = self.transform_latent_shape[0]
        else:
            self.n_segs = 32 # TODO: unhardcode

        self.segs_shape = self.conditioning_input_shapes[0][:-1] + (self.n_segs,)

    def create_modules(self):
        print('Creating CVAE with encoder params {}'.format(self.transform_enc_params))
        if self.latent_distribution == 'normal':
            self.transform_enc_model = \
                cvae_modules.transform_encoder_model(
                    input_shapes=self.ae_input_shapes,  # + [self.segs_shape],
                    input_names=self.ae_input_names,  # + ['segs'],
                    latent_shape=self.transform_latent_shape,
                    model_name='{}_transform_encoder_cvae'.format(self.transform_name),
                    enc_params=self.transform_enc_params)

            self.prior_enc_model = \
                cvae_modules.transform_encoder_model(
                    input_shapes=self.conditioning_input_shapes,# + [self.segs_shape],
                    input_names=self.conditioning_input_names,# + ['segs'],
                    latent_shape=self.transform_latent_shape,
                    model_name='prior_encoder_cvae'.format(self.transform_name),
                    enc_params=self.transform_enc_params)
        else:
            self.transform_enc_model = cvae_modules.transform_categorical_encoder_model(
                input_shapes=self.ae_input_shapes,  # + [self.segs_shape],
                input_names=self.ae_input_names,  # + ['segs'],
                latent_shape=self.transform_latent_shape,
                model_name='{}_transform_encoder_cvae'.format(self.transform_name),
                enc_params=self.transform_enc_params)

            self.prior_enc_model = \
                cvae_modules.transform_categorical_encoder_model(
                    input_shapes=self.conditioning_input_shapes,# + [self.segs_shape],
                    input_names=self.conditioning_input_names,# + ['segs'],
                    latent_shape=self.transform_latent_shape,
                    model_name='prior_encoder_cvae'.format(self.transform_name),
                    enc_params=self.transform_enc_params)

        # TODO: right now we only support segment_selector
        self.transformer_model = \
            cvae_modules.transformer_selector_model(
                conditioning_input_shapes=self.conditioning_input_shapes + [self.segs_shape],
                conditioning_input_names=self.conditioning_input_names + ['segs'],
                output_shape=self.output_shape,
                source_input_idx=self.source_im_idx,
                mask_by_conditioning_input_idx=self.mask_im_idx,
                n_segs=self.n_segs,
                latent_distribution=self.latent_distribution,
                model_name=self.transform_name + '_transformer_cvae',
                transform_type=self.transform_type,
                color_transform_type=self.color_transform_type,
                condition_on_image=self.condition_on_image,
                n_concat_scales=self.n_concat_scales,
                transform_latent_shape=self.transform_latent_shape,
                enc_params=self.dec_params,
                transform_activation=self.transform_activation,
                clip_output_range=self.clip_output_range
            )

        from keras.models import Model
        from keras.layers import Input, Activation
        seg_input = Input(self.conditioning_input_shapes[0], name='input_frame')
        segs = basic_networks.unet2D(
            seg_input,
            input_shape=self.conditioning_input_shapes[0],
            out_im_chans=self.n_segs,
            nf_enc=self.transform_enc_params['nf_enc'],
            nf_dec=self.transform_enc_params['nf_dec'],
            n_convs_per_stage=self.transform_enc_params['n_convs_per_stage'],
        )
        segs = Activation('softmax', name='softmax_segs')(segs)
        self.seg_model = Model(inputs=seg_input, outputs=segs, name='seg_model')

    def create_train_wrapper(self, n_samples=1):
        self.trainer_model = \
            cvae_modules.cvae_learned_prior_trainer_wrapper(
                ae_input_shapes=self.ae_input_shapes,
                ae_input_names=self.ae_input_names,
                conditioning_input_shapes=self.conditioning_input_shapes,
                conditioning_input_names=self.conditioning_input_names,
                output_shape=self.output_shape,
                model_name='{}_transformer_cvae_trainer'.format(self.transform_name),
                latent_distribution=self.latent_distribution,
                seg_model=self.seg_model,
                transform_encoder_model=self.transform_enc_model,
                prior_encoder_model=self.prior_enc_model,
                transformer_model=self.transformer_model,
                transform_type=self.transform_type,
                transform_latent_shape=self.transform_latent_shape,
                include_aug_matrix=self.include_aug_matrix,
                n_outputs=self.n_outputs
            )

        # TODO: include the conditional encoder if we have an AVAE
        self.tester_model = cvae_modules.cvae_learned_prior_tester_wrapper(
            conditioning_input_shapes=self.conditioning_input_shapes,
            conditioning_input_names=self.conditioning_input_names,
            latent_distribution=self.latent_distribution,
            seg_model=self.seg_model,
            prior_enc_model=self.prior_enc_model,
            dec_model=self.transformer_model,
        )
        '''
        else:
            self.trainer_model = \
                VTE_network_modules.VTE_transformer_train_model(
                    img_shape=self.img_shape,
                    output_shape=self.output_shape,
                    model_name='{}_transformer_cvae_trainer'.format(self.transform_type),
                    vte_transform_encoder_model=self.transform_enc_model,
                    vte_transformer_test_model=self.transformer_model,
                    learn_transform_mask=self.learn_transform_mask,
                    apply_flow_transform=self.transform_type == 'flow',
                    apply_color_transform=self.transform_type == 'color',
                    include_transform_loss=True,
                    color_transform_type=self.color_transform_type,
                    transform_latent_shape=self.transform_latent_shape)
            self.tester_model = self.transformer_model
        '''
        if self.latent_distribution == 'normal':
            self.vae_metrics = metrics.VAE_metrics(
                logvar_target=self.trainer_model.get_layer('latent_logvar_prior').output,
                mu_target=self.trainer_model.get_layer('latent_mean_prior').output,
                axis=-1)
        elif self.latent_distribution == 'categorical':
            self.vae_metrics = metrics.VAE_metrics_categorical(prior=self.trainer_model.get_layer('latent_categorical_prior').output, n_categories=self.n_segs)

    def get_models(self):
        return [self.transform_enc_model, self.transformer_model, self.prior_enc_model, self.seg_model]

    def _get_kl_losses(self):
        if self.latent_distribution == 'normal':
            # KL losses
            loss_names = [
                '{}_kl_mu'.format(self.transform_type), '{}_kl_logvar'.format(self.transform_type),
                'dummy_mu', 'dummy_logvar'
            ]
            loss_fns = [self.vae_metrics.kl_mu, self.vae_metrics.kl_log_sigma, 'mean_absolute_error',
                        'mean_absolute_error']
            loss_weights = [1.] * 2 + [0.] * 2
        elif self.latent_distribution == 'categorical':
            # KL losses
            loss_names = [
                '{}_kl_categorical'.format(self.transform_type),
                'dummy_prior_categorical'
            ]
            loss_fns = [self.vae_metrics.kl_categorical, 'mean_absolute_error']
            loss_weights = [1., 0]
        else:
            raise NotImplementedError('Only normal and categorical distributions are supported!')

        return loss_names, loss_fns, loss_weights

    def get_losses(self,
                   transform_reg_fn=None, transform_reg_lambda=1., transform_reg_name='lapl',
                   recon_loss_fn=None, recon_loss_weight=1., recon_loss_name='l2'):

        loss_names = ['total']
        loss_fns = []
        loss_weights = []

        # convert to list so we can consistently process multiple losses
        if not isinstance(recon_loss_fn, list):
            recon_loss_fn = [recon_loss_fn]
            recon_loss_name = [recon_loss_name]
        if not isinstance(recon_loss_weight, list):
            recon_loss_weight = [recon_loss_weight]

        if self.transform_type is not None:
            # reconstruction first since this is what we care about. then smoothness
            loss_names += [
                              '{}_recon_{}'.format(self.transform_name, rln) for rln in recon_loss_name
                          ] + [
                              '{}_smooth_{}'.format(self.transform_name, transform_reg_name),
                          ]
            loss_fns += recon_loss_fn + [transform_reg_fn]  # smoothness reg, reconstruction
            loss_weights += recon_loss_weight + [transform_reg_lambda]
        else:  # no transform, just direct synthesis
            # smoothness and reconstruction loss first
            loss_names += [
                '{}_recon_{}'.format(self.transform_name, rln) for rln in recon_loss_name
            ]
            loss_fns += recon_loss_fn  # smoothness reg, reconstruction
            loss_weights += recon_loss_weight

        # KL mean and logvar losses at end
        loss_names_kl, loss_fns_kl, loss_weights_kl = self._get_kl_losses()
        loss_names += loss_names_kl
        loss_fns += loss_fns_kl
        loss_weights += loss_weights_kl
        return loss_names, loss_fns, loss_weights

