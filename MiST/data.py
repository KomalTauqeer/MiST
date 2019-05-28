
# imports from standard python
from __future__ import print_function
import sys
# import json
import random
# imports from local packages

# imports from pip packages
# import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.utils import to_categorical

# imports from MiST
from MiST import rootIO
from MiST import globaldef as gl


pd.set_option('display.max_colwidth', -1)


class DataHandlerBase:
    """base class template for data handling"""

    def __init__(self):

        self.name = 'test name'
        self.desc = 'test description'
        self.vars = gl.arg['variables']
        self.weights = gl.arg['weights']
        self.dataobj = pd.DataFrame()

        self.filled = False

    def data(self):

        if self.dataobj.empty:
            print('WARNING: data frame should not be empty')
        else:
            return self.dataobj

    def set_name(self, name):

        self.name = name

    def set_type(self, desc):

        self.desc = desc

    def print_name(self):

        print('name: ' + self.name)

    def print_type(self):

        print('description: ' + self.desc)

    def print_variables(self):

        print('variables: ' + " ,".join(self.vars))

    def print_weights(self):

        print('weights: ' + " ,".join(self.weights))

    def print_all(self):

        self.print_name()
        self.print_type()
        self.print_variables()
        self.print_weights()


class DataHandlerTrainBinary(DataHandlerBase):
    """class to handle data from ROOT files and convert them to numpy arrays or pandas dataframes for training"""

    def __init__(self):

        DataHandlerBase.__init__(self)

    def add(self, files, tree):

        # check file argument
        if type(files) is str:
            files = [files]

        if type(files) is not list:
            print('ERROR: file list is not of type str or list!')
            sys.exit(1)

        # check tree argument
        if type(tree) is not str:
            print('ERROR: tree name is not of type str!')
            sys.exit(1)

        # check of files and tree can be read
        for ifile in files:
            rootIO.checkfile(ifile, tree)

        # load variables and weigts from file
        variables = pd.DataFrame()
        evtweights = pd.DataFrame()
        xseff = None
        for ifile in files:
            tmp_variables = get_df(ifile, tree, self.vars)
            tmp_evtweights = get_df(ifile, tree, self.weights)
            tmp_xseff = np.array([normalize(ifile, tree)] * len(tmp_variables))

            variables = merge_df(variables, tmp_variables)
            evtweights = merge_df(evtweights, tmp_evtweights)

            if xseff is None:
                xseff = tmp_xseff
            else:
                xseff = np.concatenate((xseff, tmp_xseff))

        # is this still needed?
        # idx = np.array(range(len(xseff)))

        # ================================================== modification is done here =====================================
        # Get maximum number of events per sample
        maxEvents = 219322       # This is to be corrected...



#        for ifile in files:
#            tmp_variables = get_df(ifile, tree, self.vars)
#            tmp_evtweights = get_df(ifile, tree, self.weights)
#            if maxEvents < len(tmp_variables):
#                maxEvents = len(tmp_variables)
#       print("max events are :", maxEvents)
#        print("balancing train samples.....")





        # multiply train events
        for ifile in files:
           # get factor to multiply
            tmp_variables = get_df(ifile, tree, self.vars)
            factor = 1.*maxEvents/len(tmp_variables)
            weights = evtweights.prod(axis=1)*factor 






        # ====================================================================================================================


        # multiply event weights with cross section * efficiency
        # weights = evtweights.prod(axis=1) * xseff


        # in the resulting df, variables are ordered differently ... fix this by updating variables list
        gl.arg['variables'] = list(variables)

        # combine df of variables and weights
        df = pd.concat([weights, variables], axis=1)

        self.dataobj = df
        self.filled = True


class DataHandlerEvalBinary(DataHandlerBase):
    """class to handle data from ROOT files and convert them to numpy arrays or pandas dataframes for evaluation"""

    def __init__(self):

        DataHandlerBase.__init__(self)

    def add(self, ifile, tree):

        # check tree argument
        if type(tree) is not str:
            print('ERROR: tree name is not of type str!')
            sys.exit(1)

        # check if files and tree can be read
        rootIO.checkfile(ifile, tree)

        # load variables and weights from file
        variables = get_df(ifile, tree, self.vars)

        self.dataobj = variables
        self.filled = True


class DataHandlerMultiTrain(DataHandlerTrainBinary):
    """class for multiclassification data"""

    def __init__(self):

        DataHandlerBase.__init__(self)

        self.label = -1
        self.labels = pd.DataFrame()

    def add_labels(self, label, n_outputs):

        if not self.filled:
            print('ERROR: dataobj not filled yet!')
            sys.exit(1)

        self.label = label

        datasize = self.dataobj.shape[0]

        # create N_data X 1 array with the class label
        np_label = np.full((datasize, 1), [label])

        # create N_data X N_classes array with 0 and 1
        label_cat = to_categorical(np_label,
                                   num_classes=n_outputs,
                                   dtype = int)

        self.labels = pd.DataFrame(label_cat)


class DataHandlerVariableSize(DataHandlerBase):
    """class to handle input data with varying size, for instance jets of an event"""

    def __init__(self):

        DataHandlerBase.__init__(self)


class DataHandlerTrainReco(DataHandlerBase):
    """class to handle data from ROOT files and convert them pandas dataframe,
    slight differences to DataHandlerTrainBinary due to missing weights, but additional MC truth info"""

    def __init__(self):

        DataHandlerBase.__init__(self)

        self.vars_truth = gl.arg['variables_truth']

        self.truthobj = pd.DataFrame()

    def add(self, files, tree):

        # check file argument
        if type(files) is str:
            files = [files]

        if type(files) is not list:
            print('ERROR: file list is not of type str or list!')
            sys.exit(1)

        # check tree argument
        if type(tree) is not str:
            print('ERROR: tree name is not of type str!')
            sys.exit(1)

        # check if files and tree can be read
        for ifile in files:
            rootIO.checkfile(ifile, tree)

        # load variables from file
        variables = pd.DataFrame()
        for ifile in files:
            tmp_variables = get_df(ifile, tree, self.vars)

            variables = merge_df(variables, tmp_variables)

        self.dataobj = variables

        # add truth variables from file for matching
        variables_truth = pd.DataFrame()
        for ifile in files:
            tmp_variables_truth = get_df(ifile, tree, self.vars_truth)

            variables_truth = merge_df(variables_truth, tmp_variables_truth)

        self.truthobj = variables_truth

        self.filled = True


def get_df(ifile, tree, vlist):

    if ifile == 'None':
        print('No input file!')
        sys.exit(1)

    vlist_noexpand = []
    to_flatten = []
    for v in vlist:
        if ('[' in v) and (']' in v):
            vlist_noexpand.append('noexpand:' + v)
            if not (('$' in v) or ('&&' in v) or ('||' in v)):
                to_flatten.append(v)
        elif ('(' in v) and (')' in v):
            vlist_noexpand.append('noexpand:' + v)
        else:
            vlist_noexpand.append(v)

    df = rootIO.load_df(ifile, tree, vlist_noexpand, to_flatten)

    # remove additional df column if flatten has been used
    if '__array_index' in df.columns:
        df = df.drop('__array_index', 1)

    return df


def merge_df(first, second, index_ignore=True):
    df = pd.concat((first, second), ignore_index=index_ignore)
    return df


def get_npa():
    # to be implemented...
    return 0


def merge_npa():
    # to be implemented...
    return 0


def normalize(ifile, tree):
    # this causes some problems, deactivated for the moment
    # xseff = rootIO.getcrosseff(ifile, tree)
    xseff = 1
    return xseff

def shuffle_list(*ls):

    l =list(zip(*ls))
    random.shuffle(l)

    return zip(*l)


def shuffle_npa(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def binary_labels(sig, bkg):

    df_data = merge_df(sig, bkg)
    data = df_data.values

    labels = []

    for idata, ilabel in [(sig, 1), (bkg, 0)]:
        labels.extend([ilabel] * idata.shape[0])

    labels = np.array(labels)

    data, labels = shuffle_npa(data, labels)

    return data, labels


def multi_labels(data_list):

    df_data = pd.DataFrame()
    df_labels = pd.DataFrame()

    for idata in data_list:
        df_data = merge_df(df_data, idata.data())
        df_labels = merge_df(df_labels, idata.labels)

    data = df_data.values
    labels = np.array(df_labels)

    data, labels = shuffle_npa(data, labels)

    return data, labels


def define_transform(data, path):

    scaler = StandardScaler()
    scaler.fit_transform(data)
    joblib.dump(scaler, path + '/model/scaler.joblib')


def apply_transform(data, path):

    scaler = StandardScaler()
    scaler = joblib.load(path + '/model/scaler.joblib')
    data = scaler.transform(data)
    return data


def get_weights(data):

    return data[0:, 0]

def get_variables(data):

    return data[0:, 1:]

 
