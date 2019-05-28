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

from MiST import globaldef as gl
from sklearn.metrics import confusion_matrix
"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
"""

def plot_confusion_matrix(y_true, y_pred, sample_type,
                          normalize,
                          title,
                          cmap=plt.cm.Blues):

    np.set_printoptions(precision=2)
    classes = gl.arg['multi_train_plot_labels']

    opath = gl.arg['mva_path'] + '/'

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)


    print("precision total:", precision_macro_average(cm))
    print("recall total:", recall_macro_average(cm))
    print("Accuracy:" , accuracy(cm))


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(opath + '/confusion_matrix_'+ sample_type +'.pdf')
    return ax
    
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()

def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()
def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows
def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    print('Diagonal Sum is : ' , diagonal_sum )
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

#def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    # return fbeta_score(y_true, y_pred, beta=1)

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    from sklearn.metrics import roc_auc_score
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)
                                                              
    




    
    
    
    
