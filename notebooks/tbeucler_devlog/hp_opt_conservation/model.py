import json
import keras
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm

from cbrain.losses import *
from cbrain.layers import *
from cbrain.cam_constants import *

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.backend import eval
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler
from stored_dictionaries.data_options import data_opts

class Network:
    def __init__(self, args, ID, scale_dict=None, sub=None, div=None):
        self.ID = ID
        self.args = args
        self.scale_dict = scale_dict
        self.sub = sub
        self.div = div

        self.init_paths()

        if args['alg'] == 'pbt':
            if args['load_from'] == '':
                self.build_model()
        else:
            self.build_model()

    def baseline_model(self):
        self.args['num_layers'] = 5
        for i in range(self.args['num_layers']):
            self.args['layer_%d'%i] = 512
        self.args['lr'] = 0.001
        self.args['dropout'] = 0
        self.args['batch_norm'] = False
        self.args['loss_type'] = 'mse'
        self.args['leaky_relu'] = 0.3
        self.build_model()

    def init_paths(self):
        if self.args['alg'] == 'baseline':
            self.args['results_dir'] = 'SherpaResults/baselines/{data}/{net_type}_{loss_type}/'.format(
                data=self.args['data'],
                net_type=self.args['net_type'],
                loss_type=self.args['loss_type']
            )
        else:
            self.args['results_dir'] = 'SherpaResults/{data}_{alg}/{net_type}_{loss_type}/'.format(
                data=self.args['data'],
                alg=self.args['alg'],
                net_type=self.args['net_type'],
                loss_type=self.args['loss_type']
            )
        self.args['model_dir'] = self.args['results_dir'] + 'Models/'

    def build_model(self):
        x = input = Input(shape=(data_opts[self.args['data']]['input_shape'],))
        for i in range(self.args['num_layers']):
            x = Dense(self.args['layer_%d'%i])(x)
            x = LeakyReLU(alpha=self.args['leaky_relu'])(x)

            if self.args['dropout'] != 0: x = Dropout(self.args['dropout'])(x)
            if self.args['batch_norm']: x = BatchNormalization()(x)

        if self.args['net_type'] == 'conservation':
            densout = Dense(214, activation='linear')(x)
            densout = LeakyReLU(alpha=0.3)(densout)

            surfout = SurRadLayer(
                inp_div=self.div,
                inp_sub=self.sub,
                norm_q=self.scale_dict['PHQ'],
                hyai=hyai, hybi=hybi
            )([input, densout])

            massout = MassConsLayer(
                inp_div=self.div,
                inp_sub=self.sub,
                norm_q=self.scale_dict['PHQ'],
                hyai=hyai, hybi=hybi
            )([input, surfout])

            x = EntConsLayer(
                inp_div=self.div,
                inp_sub=self.sub,
                norm_q=self.scale_dict['PHQ'],
                hyai=hyai, hybi=hybi
            )([input, massout])
        else:
            x = Dense(data_opts[self.args['data']]['output_shape'])(x)

        model = Model(inputs=input, outputs=x)

        self.model = self._compile(model, input=input)

    def _compile(self, model, input=None):
        if self.args['loss_type'] == 'weak_loss':
            loss = WeakLoss(input,
                inp_div=self.div,
                inp_sub=self.sub,
                norm_q=self.scale_dict['PHQ'],
                hyai=hyai, hybi=hybi, name='loss',
                alpha_mass=self.args['alpha']/4., alpha_ent=self.args['alpha']/4.,
                alpha_lw=self.args['alpha']/4., alpha_sw=self.args['alpha']/4.
            )
        else:
            loss = self.args['loss_type']

        model.compile(
            loss=loss,
            optimizer=Adam(lr=self.args['lr']),
            metrics=['mse']
        )
        return model

    def train(self, train_gen, valid_gen, trial=None, client=None):
        es = EarlyStopping(monitor='val_mean_squared_error',patience=self.args['patience'])
        checkpoint = ModelCheckpoint(self.get_model_path()+'.h5', save_best_only=True, save_weights_only=False)
        callbacks = [es, checkpoint]

        if trial is not None: callbacks.append(client.keras_send_metrics(trial, objective_name='val_loss', context_names=['loss', 'val_loss']))
        if self.args['alg'] == 'pbt': callbacks[-1] = client.keras_send_metrics(trial, objective_name='val_mean_squared_error', context_names=['loss', 'val_loss', 'mean_squared_error', 'val_mean_squared_error'])

        if self.args['alg'] != 'baseline':
            def schedule(epoch, lr):
                return lr * self.args['lr_decay']

            class EarlyStoppingByLossVal(Callback):
                def __init__(self, monitor='val_mean_squared_error', value=0.0025, verbose=1):
                    super(Callback, self).__init__()
                    self.monitor = monitor
                    self.value = value
                    self.verbose = verbose

                def on_epoch_end(self, epoch, logs={}):
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

                    if current > self.value:
                        if self.verbose > 0:
                            print("Epoch %05d: early stopping THR" % epoch)
                        self.model.stop_training = True

            lr_sch = LearningRateScheduler(schedule); callbacks.append(lr_sch)
            # esblv = EarlyStoppingByLossVal(monitor='val_mean_squared_error'); callbacks.append(esblv)

        # training
        if self.args['data'] == 'fluxbypass_aqua':
            history = self.model.fit_generator(
                train_gen,
                epochs=self.args['epochs'],
                validation_data=valid_gen,
                verbose=2,
                callbacks=callbacks,
            )
        else:
            history = self.model.fit_generator(
                train_gen.return_generator(),
                steps_per_epoch=train_gen.n_batches,
                epochs=self.args['epochs'],
                validation_data=valid_gen.return_generator(),
                validation_steps=valid_gen.n_batches,
                verbose=2,
           	    workers=16,
                max_queue_size=50,
                callbacks=callbacks,
            )

        with open(self.args['model_dir'] + '%05d.json' % self.ID, "w") as json_file:
            # write to file
            history.history['epochs'] = range(len(history.history['loss']))
            json_file.write(str(history.history))

    def get_model_path(self):
        return self.args['model_dir'] + '%05d' % self.ID

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.get_model_path()

        # save to h5 file
        self.model.save(file_name+'.h5')

    def load(self, file_name=None, weights_separate=False):
        if weights_separate:
            # load model structure from json file
            with open(file_name+'.json','r') as json_file:
                json_model = json_file.read()
                # create keras model from the saved json file
                model = model_from_json(json_model)
            # load weights from h5 file
            if weights: model.load_weights(file_name+'.h5')
        else:
            model = load_model(file_name)
        self.model = model
