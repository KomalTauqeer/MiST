# imports from standard python
from __future__ import print_function
# import os
# import sys

# imports from local packages
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
# imports from pip packages

# import six
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# imports from MiST
from MiST import data
from MiST import debugging
from MiST import method
from MiST import plot
from MiST import printout
from MiST import rootIO
from MiST import utilis
from MiST import globaldef as gl
from MiST import ROC_multi
from MiST import plot_confusion
from MiST import DNN_discriminator


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

    # get number of classes

    multi_train_files = gl.arg['multi_train']
    multi_train_files_class = gl.arg['multi_train_class']
    multi_train_files_plot_labels = gl.arg['multi_train_plot_labels']

    multi_test_files = gl.arg['multi_test']
    multi_test_files_class = gl.arg['multi_test_class']
    multi_test_files_plot_labels = gl.arg['multi_test_plot_labels'] 
     
    tmp_n_outputs = np.shape(multi_train_files_class)
    n_outputs = np.add.reduce(tmp_n_outputs,0)
    l_outputs = range(len(multi_train_files))
    
    # prepare the data
    print('Loading data...')

    train_data = []
    test_data = []
    
    # ========================================= modification ========================================================
   # maxEvents = 0
   # for ifile in multi_train_files:
   #     tmp_variables = data.get_df(ifile, tree, self.vars)
  #      tmp_evtweights = data.get_df(ifile, tree, self.weights)
 #       if maxEvents < len(tmp_variables):
#        maxEvents = len(tmp_variables)
   # print("max events are :", maxEvents)



    for i in l_outputs:

        train_data.append(data.DataHandlerMultiTrain())
        test_data.append(data.DataHandlerMultiTrain())

        train_data[i].add(multi_train_files[i], gl.arg['tree'])
        test_data[i].add(multi_test_files[i], gl.arg['tree'])

        train_data[i].add_labels(multi_train_files_class[i], n_outputs)
        test_data[i].add_labels(multi_test_files_class[i], n_outputs)

        train_data[i].set_name(str(multi_train_files_plot_labels[i]))
        test_data[i].set_name(str(multi_test_files_plot_labels[i]))

    

    # plotting input variables
    if do_plot_inputvars:
        plot.inputvars_multi(train_data, opath, gl.arg['variables'])

    data_train, label_train = data.multi_labels(train_data)
    data_test, label_test = data.multi_labels(test_data)

    vars_train = data.get_variables(data_train)
    vars_test = data.get_variables(data_test)

    weights_train = data.get_weights(data_train)
    weights_test = data.get_weights(data_test)

    # rootIO.save_training_inputs(vars_train, label_train, opath, gl.arg['variables'], 'bare')

    data.define_transform(vars_train, opath)
    vars_train = data.apply_transform(vars_train, opath)
    vars_test = data.apply_transform(vars_test, opath)

    # rootIO.save_training_inputs(vars_train, label_train, opath, gl.arg['variables'], 'transformed')

    print()

    n_inputs = vars_train.shape[1]

    choose_method = getattr(method, gl.arg['method'])
    model = choose_method(n_inputs = n_inputs,
                          n_outputs = n_outputs)

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
   

   # ================================================== modification ==========================================================
    print(eval_test)
    print(np.shape(eval_test))
    print("==================")
    print(label_test)
    print(label_test.shape)
    print("+++++++++++++++++")
    print(eval_train)
    print(np.shape(eval_train))
    print("//////////////////")
    print(label_train)
    print(np.shape(label_train)) 

    print("Maximum element index label_train : ",np.argmax(label_train , axis = 1))
    print("Maximum element index eval_train : ",np.argmax(eval_train , axis = 1))
    print("Maximum element index label_test : ",np.argmax(label_test, axis = 1))
    print("Maximum element index eval_test : ",np.argmax(eval_test, axis = 1))
    
    # input prepration for confusion matrix 

    true_value_train = np.argmax(label_train, axis = 1)
    predicted_value_train = np.argmax(eval_train, axis = 1)

    true_value_test = np.argmax(label_test, axis = 1)
    predicted_value_test = np.argmax(eval_test, axis = 1)

    # check

    a = np.array(true_value_train)
    unique, counts = np.unique(a, return_counts=True)
    print(dict(zip(unique, counts)))
    a = np.array(predicted_value_train)
    unique, counts = np.unique(a, return_counts=True)
    print(dict(zip(unique, counts))) 

    a = np.array(true_value_test)
    unique, counts = np.unique(a, return_counts=True)
    print(dict(zip(unique, counts)))
    a = np.array(predicted_value_test)
    unique, counts = np.unique(a, return_counts=True)
    print(dict(zip(unique, counts)))


    #=================================== plot confusion matrix======================================

#    np.set_printoptions(precision=2)
    class_names = gl.arg['multi_train_plot_labels']
   
    # Plot non-normalized confusion matrix                  # one can also implement normalization as an option

    # plot_confusion_matrix(true_value_train , predicted_value_train , classes= class_names, sample_type= 'train',
    #                   title='Confusion matrix, without normalization')
    # plot_confusion_matrix(true_value_test , predicted_value_test , classes= class_names, sample_type= 'test',
    #                   title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix

    plot_confusion.plot_confusion_matrix(true_value_train, predicted_value_train , sample_type= 'train', normalize=True,
                      title='Normalized confusion matrix')
    plot_confusion.plot_confusion_matrix(true_value_test, predicted_value_test , sample_type= 'test', normalize=True,
                      title='Normalized confusion matrix')


    # ========================================================== F1 score =================================================


    from sklearn.metrics import f1_score
    f1_score_train = f1_score(true_value_train, predicted_value_train , average = 'micro')   # one can also implement to make average option as users choice
    f1_score_test = f1_score(true_value_test, predicted_value_test , average = 'micro')
    print(" F1 score for your classifier ")
    print(" For train sample :  " ,f1_score_train)
    print(" For test sample :  " ,f1_score_test)

    # ==================================================== ROC AUC Score ===================================================



    from sklearn.preprocessing import label_binarize
    y = label_binarize(predicted_value_train,class_names)
    x = label_binarize(true_value_train,class_names)
    print('multi-class-ROC_AUC :', plot_confusion.multiclass_roc_auc_score(true_value_test,predicted_value_test))



    # ======================================================= ROC Curves ===================================================
    
    # Plotting ROC curves for each class
    ROC_multi.roc_curves_multi(label_test,eval_test,n_outputs)


    # ===================================================== DNN Output plots ===============================================

    DNN_discriminator.plotDiscriminators(label_test,eval_test,sample = 'test')

    # ==========================================================================================================================


    rootIO.save_multi_training_reults(label_train=label_train,
                                      eval_train=eval_train,
                                      label_test=label_test,
                                      eval_test=eval_test,
                                      n_outputs=n_outputs,
                                      opath=opath)

    # save the model and weights to files
    print('--- Saving model')
    model.save()

    print('\n\n')
    print('--- Finished training!')



    # ============================================ modifictaion ================================================================


#def plot_confusion_matrix(y_true, y_pred, classes, sample_type,
#                          normalize=False,
#                          title=None,
#                          cmap=plt.cm.Blues):
#
#    
#    opath = gl.arg['mva_path'] + '/'
#
#    from sklearn import svm, datasets
#    from sklearn.model_selection import train_test_split
#    from sklearn.metrics import confusion_matrix
#    from sklearn.utils.multiclass import unique_labels
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if not title:
#        if normalize:
#            title = 'Normalized confusion matrix'
#        else:
#            title = 'Confusion matrix, without normalization'
#
#    # Compute confusion matrix
#
#    cm = confusion_matrix(y_true, y_pred)
#    # Only use the labels that appear in the data
#    # classes = classes[unique_labels(y_true, y_pred)]
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#
#    print("precision total:", precision_macro_average(cm))
#    print("recall total:", recall_macro_average(cm))
#    print("Accuracy:" , accuracy(cm))
#    
#
#    fig, ax = plt.subplots()
#    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#    ax.figure.colorbar(im, ax=ax)
#    # We want to show all ticks...
#    ax.set(xticks=np.arange(cm.shape[1]),
#           yticks=np.arange(cm.shape[0]),
#           # ... and label them with the respective list entries
#           xticklabels=classes, yticklabels=classes,
#           title=title,
#           ylabel='True label',
#           xlabel='Predicted label')
#
#    # Rotate the tick labels and set their alignment.
#    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")
#
#    # Loop over data dimensions and create text annotations.
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i in range(cm.shape[0]):
#        for j in range(cm.shape[1]):
#            ax.text(j, i, format(cm[i, j], fmt),
#                    ha="center", va="center",
#                    color="white" if cm[i, j] > thresh else "black")
#    fig.tight_layout()
#    plt.savefig(opath + '/confusion_matrix_'+ sample_type +'.pdf')
#    return ax
#
#
#
#def precision(label, confusion_matrix):
#    col = confusion_matrix[:, label]
#    return confusion_matrix[label, label] / col.sum()
#    
#def recall(label, confusion_matrix):
#    row = confusion_matrix[label, :]
#    return confusion_matrix[label, label] / row.sum()
#def precision_macro_average(confusion_matrix):
#    rows, columns = confusion_matrix.shape
#    sum_of_precisions = 0
#    for label in range(rows):
#        sum_of_precisions += precision(label, confusion_matrix)
#    return sum_of_precisions / rows
#def recall_macro_average(confusion_matrix):
#    rows, columns = confusion_matrix.shape
#    sum_of_recalls = 0
#    for label in range(columns):
#        sum_of_recalls += recall(label, confusion_matrix)
#    return sum_of_recalls / columns
#def accuracy(confusion_matrix):
#    diagonal_sum = confusion_matrix.trace()
#    print('Diagonal Sum is : ' , diagonal_sum )
#    sum_of_all_elements = confusion_matrix.sum()
#    return diagonal_sum / sum_of_all_elements
#
##def fmeasure(y_true, y_pred):
#    # Calculates the f-measure, the harmonic mean of precision and recall.
#    # return fbeta_score(y_true, y_pred, beta=1)
#
#def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
#    from sklearn.metrics import roc_auc_score    
#    from sklearn import preprocessing
#    lb = preprocessing.LabelBinarizer()
#    lb.fit(y_test)
#
#    y_test = lb.transform(y_test)
#    y_pred = lb.transform(y_pred)
#
#    return roc_auc_score(y_test, y_pred, average=average)
