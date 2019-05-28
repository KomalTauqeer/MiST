# imports from standard python
from __future__ import print_function
# import os
# import sys

# imports from local packages
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
# imports from pip packages

# import six

# imports from MiST
from MiST import data
from MiST import debugging
from MiST import method
from MiST import plot
from MiST import printout
from MiST import rootIO
from MiST import utilis
from MiST import globaldef as gl


def init():

    # TODO: implement correctly
    do_plot_model = True

    # not needed, but makes it nicer
    opath = gl.arg['mva_path'] + '/'

    utilis.training_path(opath)

    # write hash
    utilis.write_hash()

    do_plot_inputvars = gl.arg['plotvars']
    # override for testing
    do_plot_inputvars = True
    debug = gl.arg['debug']

    # prepare the data
    print('Loading data...')
    train_sig = data.DataHandlerTrainBinary()
    train_bkg = data.DataHandlerTrainBinary()
    test_sig = data.DataHandlerTrainBinary()
    test_bkg = data.DataHandlerTrainBinary()

    train_sig.add(gl.arg['signal'], gl.arg['tree'])
    train_bkg.add(gl.arg['background'], gl.arg['tree'])
    test_sig.add(gl.arg['signal_test'], gl.arg['tree'])
    test_bkg.add(gl.arg['background_test'], gl.arg['tree'])

    # plotting input variables
    if do_plot_inputvars:
        plot.inputvars(train_sig.data(), train_bkg.data(), opath, gl.arg['variables'])

    # merge into single df
    # df_all = datahandler.merge(df_sig,df_bkg)

    # create labels for the data
    # label_all = datahandler.labels_binary(df_all, df_sig, df_bkg)

    # turn data into matrix
    # data_all = df_all.as_matrix()

    if debug:
        debugging.inputdata(train_sig.data(), train_bkg.data())

    data_train, label_train = data.binary_labels(train_sig.data(), train_bkg.data())
    data_test, label_test = data.binary_labels(test_sig.data(), test_bkg.data())

    vars_train = data.get_variables(data_train)
    vars_test = data.get_variables(data_test)

    weights_train = data.get_weights(data_train)
    weights_test = data.get_weights(data_test)

    rootIO.save_training_inputs(vars_train, label_train, opath, gl.arg['variables'], 'bare')

    data.define_transform(vars_train, opath)
    vars_train = data.apply_transform(vars_train, opath)
    vars_test = data.apply_transform(vars_test, opath)

    rootIO.save_training_inputs(vars_train, label_train, opath, gl.arg['variables'], 'transformed')

    print()

    if debug:
        debugging.data(vars_train, label_train, vars_test, label_test)

    printout.data(train_sig.data(), train_bkg.data(), test_sig.data(), test_bkg.data(), label_train, label_test)

    n_inputs = vars_train.shape[1]

    # implement weights for the trainingsamples here

    # train_weights = tf.placeholder(name="loss_weights", shape=[None], dtype=tf.float32)

    choose_method = getattr(method, gl.arg['method'])
    model = choose_method(n_inputs=n_inputs)

    model.show()

    print('Training...')

    model.train(data_train=vars_train,
                label_train=label_train,
                weights_train=weights_train,
                data_test=vars_test,
                label_test=label_test,
                weights_test=weights_test)

    print('Training finished!')

    model.score(data_test=vars_test,
                label_test= label_test)

    print('\nEvaluating training sample...')
    eval_train = model.eval(vars_train)
    print()
    print('\nEvaluating test sample...')
    eval_test = model.eval(vars_test)

    print('\n')

    utilis.roc(label_train, eval_train, opath, 'test')
    utilis.roc(label_test, eval_test, opath, 'train')
    plot.overtrain(eval_train, label_train, eval_test, label_test, opath)

    rootIO.save_training_reults(label_train=label_train,
                                eval_train=eval_train,
                                label_test=label_test,
                                eval_test=eval_test,
                                opath=opath)

    # save the model and weights to files
    print('--- Saving model')
    model.save()

    print('\n\n')
    print('--- Finished training!')
