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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class Network:
    def __init__(self, args):
        self.args = args
        if 'hyper_param_opt' in args['run_type']:
            self.build_model()
        elif args['run_type'] == 'baseline':
            self.baseline_model()

    def baseline_model(self):
        self.args['num_layers'] = 5
        for i in range(self.args['num_layers']):
            self.args['layer_%d'%i] = 512
        self.args['lr'] = 0.001
        self.args['dropout'] = 0
        self.args['batch_norm'] = False
        self.args['loss'] = 'mse'
        self.args['leaky_relu'] = 0.3
        self.build_model()

    def build_model(self):
        x = input = Input(shape=(304,))
        for i in range(self.args['num_layers']):
            x = Dense(self.args['layer_%d'%i])(x)
            x = LeakyReLU(alpha=self.args['leaky_relu'])(x)

            if self.args['dropout'] != 0: x = Dropout(self.args['dropout'])(x)
            if self.args['batch_norm']: x = BatchNormalization()(x)

        if self.args['conservation']:
            densout = Dense(214, activation='linear')(densout)
            densout = LeakyReLU(alpha=0.3)(densout)

            surfout = SurRadLayer(
                inp_div=self.args['div'],
                inp_sub=self.args['sub'],
                norm_q=scale_dict['PHQ'],
                hyai=hyai, hybi=hybi
            )([input, densout])

            massout = MassConsLayer(
                inp_div=self.args['div'],
                inp_sub=self.args['sub'],
                norm_q=scale_dict['PHQ'],
                hyai=hyai, hybi=hybi
            )([input, surfout])

            x = EntConsLayer(
                inp_div=self.args['div'],
                inp_sub=self.args['sub'],
                norm_q=self.args['scale_dict']['PHQ'],
                hyai=hyai, hybi=hybi
            )([input, massout])
        else:
            x = Dense(218)(x)

        model = Model(inputs=input, outputs=x)

        self.model = self._compile(model, input=input)

    def _compile(self, model, input=None):
        if self.args['loss'] == 'weak_loss':
            loss = WeakLoss(input,
                inp_div=self.args['div'],
                inp_sub=self.args['sub'],
                norm_q=self.args['scale_dict']['PHQ'],
                hyai=hyai, hybi=hybi, name='loss',
                alpha_mass=self.args['alpha'], alpha_ent=self.args['alpha'],
                alpha_lw=self.args['alpha'], alpha_sw=self.args['alpha']
            )
        else:
            loss = self.args['loss']

        model.compile(
            loss=loss,
            optimizer=Adam(lr=self.args['lr']),
            #metrics=['mse']
        )
        return model

    def train(self, train_gen, valid_gen, trial=None, study=None):
        es = EarlyStopping(monitor='val_loss',patience=self.args['patience'])
        checkpoint = ModelCheckpoint(self.get_model_path()+'.h5', save_best_only=True, save_weights_only=True)
        callbacks = [es, checkpoint]

        if trial is not None: callbacks.append(study.keras_callback(trial, objective_name='val_loss'))

        # training
        history = self.model.fit_generator(
            train_gen,
            epochs=self.args['epochs'],
            validation_data=valid_gen,
            verbose=1,
            callbacks=callbacks,
        )

        if study is not None: study.finalize(trial)

        return history.history

    def predict(self, gen, file_name):
        predictions = self.model.predict_generator(gen)
        np.savez(file_name + '.npz', predictions=predictions)

    def get_model_path(self):
        file_name = self.args['model_dir'] + 'best_model'
        if 'hyper_param_opt' in self.args['run_type']:
            file_name = self.args['model_dir'] + '%05d' % self.args['trial_count']
        return file_name

    def save(self, file_name=None, weights=False):
        if file_name is None:
            file_name = self.get_model_path()

        # save model structure to json file
        with open(file_name+'.json', "w") as json_file:
            json_model = self.model.to_json()
            # get learning rate of model and store in json
            json_model = json_model[0] + "\"lr\": {},".format(eval(self.model.optimizer.lr)) + json_model[1:]
            # write to file
            json_file.write(json_model)

        # save weights to h5 file
        if weights: self.model.save_weights(file_name+'.h5')

    def load(self, file_name=None, weights=False):
        # if no file specified use the best model from hyper param trials
        if file_name is None: file_name = self.args['model_location']
        # load model structure from json file
        with open(file_name+'.json','r') as json_file:
            json_model = json_file.read()
            # set lr param in args from model dump
            self.args['lr'] = json.loads(json_model)['lr']
            # create keras model from the saved json file
            model = model_from_json(json_model)
        # load weights from h5 file
        if weights: model.load_weights(file_name+'.h5')

        self.model = self._compile(model)
