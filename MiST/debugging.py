# imports from standard python
from __future__ import print_function

# imports from local packages

# imports from pip packages
import numpy as np
from pympler import summary
from pympler import muppy

# imports from MiST


def inputdata(df_sig, df_bkg):
    print('df_sig: ' + str(len(df_sig)))
    print(df_sig)
    print('df_bkg: ' + str(len(df_bkg)))
    print(df_bkg)


def data(data_train, label_train, data_test, label_test):
    print('data_train: ' + str(len(data_train)))
    print(data_train)
    print('data_test: ' + str(len(data_test)))
    print(data_test)
    print('label_train: ' + str(len(label_train)) + ' ( ' + str(np.count_nonzero(label_train == 1)) + ' signal / ' + str(np.count_nonzero(label_train == 0)) + ' background )')
    print(label_train)
    print('label_test: ' + str(len(label_test)) + ' ( ' + str(np.count_nonzero(label_test == 1)) + ' signal / ' + str(np.count_nonzero(label_test == 0)) + ' background )')
    print(label_test)


def mem():
    all_objects = muppy.get_objects()
    summary.print_(summary.summarize(all_objects))
