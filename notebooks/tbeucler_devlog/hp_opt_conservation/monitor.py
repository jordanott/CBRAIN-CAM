import os
import sherpa
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('agg')
import matplotlib.pyplot as plt

class MetricMonitor:

    def __init__(self, args):
        self.args = args

        self.init_paths()

        if 'hyper_param_opt' in args['run_type']:
            parameters = [
                sherpa.Continuous('alpha', [0., 1]),
                sherpa.Continuous('dropout', [0., 0.5]),
                sherpa.Continuous('lr', [0.00001, 0.01]),
                sherpa.Continuous('leaky_relu', [0., 0.5]),
                sherpa.Ordinal('batch_norm', [True, False]),
                sherpa.Ordinal('loss', ['mse', 'weak_loss']),
                sherpa.Discrete('num_layers', [1, self.args['max_dense_layers']]),
            ]

            parameters.extend([
                sherpa.Discrete('layer_{}'.format(i), [32, 512]) for i in range(self.args['max_dense_layers'])
            ])

            default_params = {
            	'num_layers':5,
            	'layer_0': 512,
            	'layer_1': 512,
            	'layer_2': 512,
            	'layer_3': 512,
            	'layer_4': 512,
                'layer_5': 512,
                'layer_6': 512,
                'layer_7': 512,
                'layer_8': 512,
            	'alpha': .1,
            	'loss': 'weak_loss',
            	'dropout': 0.25,
            	'batch_norm': True,
            	'leaky_relu': .3,
                'lr': 0.001,
            }

            if args['run_type'] == 'hyper_param_opt_conservation':
                default_params.update({'conservation':True})
                parameters.append(sherpa.Ordinal('conservation', [True, False]))

            alg = sherpa.algorithms.LocalSearch(default_params) # max_num_trials=100)
            self.study = sherpa.Study(parameters=parameters,algorithm=alg,lower_is_better=True,output_dir='Params')

            self.init_trial_storage()
        elif args['run_type'] == 'baseline':
            self.init_trial_storage()

    def init_paths(self):
        self.args['results_dir'] = 'Results/{data}/{run_type}/'.format(
            data=self.args['data'],
            run_type=self.args['run_type']
        )
        self.args['model_dir'] = self.args['results_dir'] + 'Models/'
        self.args['predictions_dir'] = self.args['data_dir'] + self.args['data'] + '/Predictions/' + self.args['run_type'] + '/'

        self.build_directory(self.args['model_dir'])
        self.build_directory(self.args['predictions_dir'])

    def build_directory(self, path, current_path=''):
        # iterate through folders in specifide path
        for folder in path.split('/'):
            current_path += folder +'/'
            # if it doesn't exist build that director
            if not os.path.exists(current_path):
                os.mkdir(current_path)

    def run_hyper_param_opt(self):
        # iterate through trials in study
        for trial in self.study:
            self.args.update(trial.parameters)
            yield self.args, (trial, self.study)

    def init_trial_storage(self):
        if self.args['continue']:
            self.trial_results = pd.read_csv(self.args['results_dir']+'in_progress_training_results.csv', index_col=0)
            self.args['trial_count'] = self.trial_results['trial'].max() + 1
            print('Starting from checkpoint... Trial count:', self.args['trial_count'])
        else:
            self.args['trial_count'] = 1
            columns = [
                'trial', 'loss', 'val_loss'
            ]
            self.trial_results = pd.DataFrame(columns=columns)

    def update_trial_storage(self, history):
        # create blank holder for results
        trial = pd.DataFrame(history)

        # record results from trial
        trial['epochs'] = range(len(history['loss']))
        trial['trial'] = self.args['trial_count']

        # add to master storage
        self.trial_results = self.trial_results.append(trial, ignore_index=True)

        print('Trial: {}, Loss: {}, Val Loss: {}, Num Layers: {}, Best Val Loss: {}, Best Trial: {}'.format(
            self.args['trial_count'],
            trial['loss'].min(),
            trial['val_loss'].min(),
            self.args['num_layers'],
            self.trial_results['val_loss'].min(),
            self.trial_results['val_loss'].argmin()
        ))

        self.save(in_progress='in_progress_')

        # check if best results
        return trial['val_loss'].min() == self.trial_results['val_loss'].min()

    def end_trial(self):
    	self.args['trial_count'] += 1

    def save(self, results_dir=None, in_progress=''):
        if results_dir is None: results_dir = self.args['results_dir']
        # save df to csv
        self.trial_results.to_csv(results_dir+in_progress+'training_results.csv')

    def get_pred_loc(self, best_results=True):
        name = 'best_results' if best_results else '%05d' % self.args['trial_count'] - 1
        return self.args['predictions_dir'] + name
