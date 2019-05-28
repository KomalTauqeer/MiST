# imports from standard python
from __future__ import print_function
import ConfigParser as configparser
import sys

# imports from local packages

# imports from pip packages
from keras import optimizers
from keras.models import Sequential, load_model
# from keras.models import model_from_json, model_from_yaml
from keras.layers import Highway, MaxoutDense, Dense, Activation, Dropout
from keras.utils import multi_gpu_model, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from sklearn import tree as dtree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.externals import joblib

# imports from MiST
from MiST import globaldef as gl
from MiST import plot
from MiST import utilis

'''
different classes for different MVA methods.
in order to have a consistent train/eval module, each MVA method should have the following class methods:

show: print structure of method, parameters, etc.
train: train the method
score: get performance of the training
eval: apply method to testing data set
save: save the trained method for later use
load: load the trained method from a file for eval

if some are not needed, they should be implemented as dummies
'''


class KerasBase:

    def __init__(self):
        self.name = 'name'
        self.desc = 'description'

        self.modelpath = gl.arg['mva_path'] + '/model/'
        self.modelcheckpoint_str = '/weights_e-{epoch:03d}_l-{val_loss:.3f}_a-{val_acc:.3f}.h5'
        self.tbpath = gl.arg['mva_path'] + '/TensorBoard/'

        opt_config = configparser.ConfigParser()
        opt_config_file = './config/method/keras_optimizer.config'
        opt_config.read(opt_config_file)

        try:
            opt_str = opt_config.get('optimizer', 'optimizer')
        except configparser.NoSectionError:
            print('ERROR: could not find optimizer string in config file: ' + opt_config_file)
            sys.exit(1)

        if not hasattr(optimizers, opt_str):
            print('ERROR: could not find optimizer: ' + opt_str)
            sys.exit(1)

        try:
            list_from_config = opt_config.items(opt_str)
        except configparser.NoSectionError:
            print('ERROR: could not find optimizer section in config file: ' + opt_config_file)
            sys.exit(1)

        # convert strings to right data type and return a dict
        opt_settings = utilis.convert_list_to_dict(list_from_config)

        setup_opt = getattr(optimizers, opt_str)
        self.optimizer = setup_opt(**opt_settings)


class KerasBinary(KerasBase):

    def __init__(self, n_inputvars):

        KerasBase.__init__(self)

        self.model = Sequential()

        self.multimodel = None
        self.n_inputvars = n_inputvars
        self.n_outputs = 1
        self.is_first_layer = True
        self.multigpu = False

        self.batchsize = gl.arg['batchsize']
        self.multibatchsize = self.batchsize * gl.arg['gpucores']
        self.epochs = gl.arg['epochs']
        self.validation_fraction = gl.arg['validationsize']

        self.metrics = ['accuracy']
        self.loss = 'binary_crossentropy'

    def add_layer(self, n_nodes, activation):

        if self.is_first_layer:
            self.model.add(Dense(n_nodes,
                                 input_dim=self.n_inputvars,
                                 activation=activation))
            self.is_first_layer = False

        else:
            self.model.add(Dense(n_nodes,
                                 activation=activation))

    def add_output(self, activation):

        self.model.add(Dense(self.n_outputs,
                             activation=activation))

    def add_dropout(self, dropoutfraction):

        self.model.add(Dropout(dropoutfraction))

    def compile(self):

        if self.multigpu:

            self.multimodel = multi_gpu_model(self.model)
            self.multimodel.compile(loss=self.loss,
                                    optimizer=self.optimizer,
                                    metrics=self.metrics)

        else:
            self.model.compile(loss=self.loss,
                               optimizer=self.optimizer,
                               metrics=self.metrics)

    def show(self):

        self.model.summary()
        self.plot()

    def enable_multi_gpu(self):

        self.multigpu = True

    def get_model(self):

        return self.model

    def get_multimodel(self):

        if self.multimodel is not None:
            return self.multimodel
        else:
            print('ERROR: no multimodel available!')

    def train(self, **kwargs):

        data_train = kwargs['data_train']
        label_train = kwargs['label_train']
        weights_train = kwargs['weights_train']
        data_test = kwargs['data_test']
        label_test = kwargs['label_test']
        weights_test = kwargs['weights_test']

        if self.multigpu:
            training = self.multimodel.fit(data_train,
                                           label_train,
                                           callbacks=[
                                               EarlyStopping(verbose=True,
                                                             patience=30,
                                                             monitor='val_loss')],
                                           epochs=self.epochs,
                                           batch_size=self.multibatchsize,
                                           validation_split=self.validation_fraction,
                                           sample_weight=weights_train)

        else:
            training = self.model.fit(data_train,
                                      label_train,
                                      callbacks=[
                                          EarlyStopping(verbose=True,
                                                        patience=50,
                                                        monitor='val_loss'),
                                          ReduceLROnPlateau(monitor='val_loss',
                                                            factor=0.5,
                                                            verbose=0,
                                                            patience=10,
                                                            mode='auto',
                                                            min_delta=0.0001,
                                                            cooldown=0,
                                                            min_lr=0),
                                          ModelCheckpoint(self.modelpath + self.modelcheckpoint_str,
                                                          monitor='val_loss',
                                                          verbose=True,
                                                          save_best_only=True),
                                          TensorBoard(log_dir=self.tbpath,
                                                      histogram_freq=0,
                                                      batch_size=32,
                                                      write_graph=True,
                                                      write_grads=False,
                                                      write_images=False,
                                                      embeddings_freq=0,
                                                      embeddings_layer_names=None,
                                                      embeddings_metadata=None,
                                                      embeddings_data=None,
                                                      update_freq='epoch')],
                                      epochs=self.epochs,
                                      batch_size=self.batchsize,
                                      validation_split=self.validation_fraction,
                                      #validation_data=(data_test, label_test, weights_test),
                                      sample_weight=weights_train)

            utilis.add_tb_link()

        plot.training_history(training.history, gl.arg['mva_path'])

    def plot(self):

        plot_model(self.model,
                   to_file=gl.arg['mva_path'] + '/model.png',
                   show_shapes=True,
                   show_layer_names=True)

    def score(self, data_test, label_test):

        print('\nScore..')

        if self.multigpu:
            score = self.multimodel.evaluate(data_test,
                                             label_test,
                                             batch_size=self.multibatchsize)
        else:
            score = self.model.evaluate(data_test,
                                        label_test,
                                        batch_size=self.batchsize)

        print()
        print('test_loss: {:f} - test_acc: {:f}'.format(score[0], score[1]))

    def eval(self, data):

        return self.model.predict(data,
                                  verbose=True,
                                  batch_size=self.batchsize)

    def save(self):

        self.model.save(self.modelpath + 'complete.h5')
        self.model.save_weights(self.modelpath + 'weights.h5')
        model_json = self.model.to_json()
        with open(self.modelpath + 'model.json', 'w') as json_file:
            json_file.write(model_json)
        model_yaml = self.model.to_yaml()
        with open(self.modelpath + 'model.yaml', 'w') as yaml_file:
            yaml_file.write(model_yaml)

    def load(self):

        model_path = gl.arg['mva_path'] + '/model/complete.h5'
        self.model = load_model(model_path)

        return self.model


class KerasMulti(KerasBinary):

    def __init__(self, n_inputvars, n_outputs):

        KerasBinary.__init__(self, n_inputvars)

        self.n_outputs = n_outputs
        self.loss = 'categorical_crossentropy'


class SklearnBase:

    def __init__(self):
        self.name = 'name'
        self.desc = 'description'

        self.modelpath = gl.arg['mva_path'] + '/model/'
        self.modelfilename = 'sklearn_model.joblib'

    def load_config(self, fname):

        config = configparser.ConfigParser()
        config.read('./config/method/' + fname + '.config')

        return config


class SklearnDT(SklearnBase):

    def __init__(self):

        SklearnBase.__init__(self)

        self.classifier = dtree.DecisionTreeClassifier(criterion='gini',
                                                       splitter='best',
                                                       max_depth=None,
                                                       min_samples_split=2,
                                                       min_samples_leaf=1,
                                                       min_weight_fraction_leaf=0.0,
                                                       max_features=None,
                                                       random_state=None,
                                                       max_leaf_nodes=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       class_weight=None,
                                                       presort=False)

    def show(self):

        dt_options = self.classifier.get_params()

        print('-' * gl.screenwidth)
        print('--- Decision tree options:')
        print('-' * gl.screenwidth)
        for i in dt_options:
            print('--- {:30s} {:s}'.format(i, str(dt_options[i])))
        print()
        print('-' * gl.screenwidth)

    def train(self, **kwargs):

        data_train = kwargs['data_train']
        label_train = kwargs['label_train']

        self.classifier.fit(data_train, label_train)

    def score(self, **kwargs):

        data_test = kwargs['data_test']
        label_test = kwargs['label_test']

        score = self.classifier.score(data_test, label_test)

        print('mean accuracy: ' + str(score))

    def eval(self, data):

        return self.classifier.predict(data)

    def save(self):

        joblib.dump(self.classifier, self.modelpath + self.modelfilename)

    def load(self):

        model_path = self.modelpath + self.modelfilename
        self.classifier = joblib.load(model_path)

        return self.classifier


class SklearnBDT(SklearnDT):

    def __init__(self):

        SklearnDT.__init__(self)

        self.boosting = 'adaptive'

        self.config = self.load_config('bdt')

        self.classifier = AdaBoostClassifier(dtree.DecisionTreeClassifier(criterion=self.config.get('decision tree', 'criterion'),
                                                                          splitter=self.config.get('decision tree', 'splitter'),
                                                                          max_depth=self.config.getint('decision tree', 'max_depth'),
                                                                          min_samples_split=self.config.getint('decision tree', 'min_samples_split'),
                                                                          min_samples_leaf=self.config.getint('decision tree', 'min_samples_leaf'),
                                                                          min_weight_fraction_leaf=0.0,
                                                                          max_features=None,
                                                                          random_state=None,
                                                                          max_leaf_nodes=None,
                                                                          min_impurity_decrease=0.0,
                                                                          min_impurity_split=None,
                                                                          class_weight=None,
                                                                          presort=False),
                                             algorithm=self.config.get('adaboost', 'algorithm'),
                                             n_estimators=int(self.config.getint('adaboost', 'n_estimators')),
                                             learning_rate=self.config.getfloat('adaboost', 'learning_rate'),
                                             random_state=None)

    def show(self):

        dt_options = self.classifier.get_params()

        print('-' * gl.screenwidth)
        print('--- Boosted decision tree options: ' + self.boosting)
        print('-' * gl.screenwidth)
        for i in dt_options:
            if i is not 'base_estimator':
                print('--- {:50s} {:s}'.format(i, str(dt_options[i])))
        print('-' * gl.screenwidth)

    def eval(self, data):

        return self.classifier.decision_function(data)


class SklearnBDTG(SklearnBDT):

    def __init__(self):

        SklearnBDT.__init__(self)

        self.boosting = 'gradient'

        self.config = self.load_config('bdtg')

        self.classifier = GradientBoostingClassifier(loss=self.config.get('gradient', 'loss'),
                                                     learning_rate=self.config.getfloat('gradient', 'learning_rate'),
                                                     n_estimators=self.config.getint('gradient', 'n_estimators'),
                                                     subsample=self.config.getfloat('gradient', 'subsample'),
                                                     criterion=self.config.get('gradient', 'criterion'),
                                                     min_samples_split=self.config.getint('gradient', 'min_samples_split'),
                                                     min_samples_leaf=self.config.getint('gradient', 'min_samples_leaf'),
                                                     min_weight_fraction_leaf=self.config.getfloat('gradient', 'min_weight_fraction_leaf'),
                                                     max_depth=self.config.getint('gradient', 'max_depth'),
                                                     min_impurity_decrease=self.config.getfloat('gradient', 'min_impurity_decrease'),
                                                     min_impurity_split=None,
                                                     init=None,
                                                     random_state=None,
                                                     max_features=None,
                                                     verbose=self.config.getint('gradient', 'verbose'),
                                                     max_leaf_nodes=None,
                                                     warm_start=False,
                                                     presort='auto',
                                                     validation_fraction=self.config.getfloat('gradient', 'validation_fraction'),
                                                     n_iter_no_change=None,
                                                     tol=self.config.getfloat('gradient', 'tol'))


class SklearnBDTB(SklearnBDT):

    def __init__(self):

        SklearnBDT.__init__(self)

        self.boosting = 'bagging'
        self.classifier = BaggingClassifier(dtree.DecisionTreeClassifier(criterion='gini',
                                                                         splitter='best',
                                                                         max_depth=1,
                                                                         min_samples_split=2,
                                                                         min_samples_leaf=1,
                                                                         min_weight_fraction_leaf=0.0,
                                                                         max_features=None,
                                                                         random_state=None,
                                                                         max_leaf_nodes=None,
                                                                         min_impurity_decrease=0.0,
                                                                         min_impurity_split=None,
                                                                         class_weight=None,
                                                                         presort=False),
                                            n_estimators=200,
                                            max_samples=1.0,
                                            max_features=1.0,
                                            bootstrap=True,
                                            bootstrap_features=False,
                                            oob_score=False,
                                            warm_start=False,
                                            n_jobs=None,
                                            random_state=None,
                                            verbose=0)


''' methods called from train/eval below '''


def build_network(model, multigpu=False):

    allowed_relu_names = ['relu']
    allowed_sigmoid_names = ['sig','sigmoid']
    allowed_softmax_names = ['sm','softmax']
    allowed_dropout_names = ['do','dropout']

    if gl.arg['topology'] == None:
        print('--- you have chosen to use a DNN, but did not specify the layout via \'topology\'!')
        sys.exit(1)

    layout = gl.arg['topology']

    # decipher structure of the network
    for layer in layout:

        l_n = layer.split(':')[0].lower()
        l_o = layer.split(':')[1]

        if l_n in allowed_relu_names:
            model.add_layer(int(l_o), 'relu')

        if l_n in allowed_sigmoid_names:
            model.add_layer(int(l_o), 'sigmoid')

        if l_n in allowed_softmax_names:
            model.add_layer(int(l_o), 'softmax')

        if l_n in allowed_dropout_names:
            model.add_dropout(float(l_o))

    if multigpu:
        print('--- enabled multigpu support, still experimental!')
        model.enable_multi_gpu()

    model.compile()

    return model


def dnn(**kwargs):

    m = KerasBinary(kwargs['n_inputs'])

    return build_network(m)


def dnn_parallel(**kwargs):

    m = KerasBinary(kwargs['n_inputs'])

    return build_network(m, True)


def dnn_multi(**kwargs):

    m = KerasMulti(kwargs['n_inputs'], kwargs['n_outputs'])

    return build_network(m)


def dt(**kwargs):

    m = SklearnDT()

    return m


def bdt(**kwargs):

    m = SklearnBDT()

    return m


def bdtg(**kwargs):

    m = SklearnBDTG()

    return m

'''
def bdtb(**kwargs):
    n_trees = 200

    m = SklearnBDTB(n_trees)

    return m
'''

'''
def dnn_binary_example(n_inputs):
    model = Sequential()
    model.add(Dense(64, input_dim=n_inputs, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=keras_optimizer(),
                  metrics=['accuracy'])

    return model

def ann(n_inputs):
    model = Sequential()
    model.add(Dense(n_inputs+1, input_dim=n_inputs, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras_optimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def dnn_simple(n_inputs):
    model = Sequential()
    model.add(Dense(30, input_dim=n_inputs, activation='relu', name='layer1'))
    model.add(Dropout(0.5))
    model.add(Dense(30, activation='relu', name='layer2'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name='layer3'))

    model.compile(optimizer=keras_optimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def dnn_simple_no_dropout(n_inputs):
    model = Sequential()
    model.add(Dense(30, input_dim=n_inputs, activation='relu', name='layer1'))
    model.add(Dense(30, activation='relu', name='layer2'))
    model.add(Dense(1, activation='sigmoid', name='layer3'))

    model.compile(optimizer=keras_optimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def dnn_simple_low_dropout(n_inputs):
    model = Sequential()
    model.add(Dense(30, input_dim=n_inputs, activation='relu', name='layer1'))
    model.add(Dropout(0.3))
    model.add(Dense(30, activation='relu', name='layer2'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', name='layer3'))

    model.compile(optimizer=keras_optimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def dnn(n_inputs):
    model = Sequential()
    model.add(Dense(100, input_dim=n_inputs, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras_optimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model





# this model is only for comparison with thee DNN from TMVA in the s channel analysis
def dnn_tmva(n_inputs):
    model = Sequential()
    model.add(Dense(100, input_dim=n_inputs, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='tanh'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear',))

    model.compile(optimizer=keras_optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
'''
