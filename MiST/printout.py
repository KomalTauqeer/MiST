# imports from standard python
from __future__ import print_function

# imports from local packages

# imports from pip packages

# imports from MiST


def data(df_sig, df_bkg, df_sig_test, df_bkg_test, label_train, label_test):
    print('Number of signal events for training:        {}'.format(len(df_sig)))
    print('Number of background events for training:    {}'.format(len(df_bkg)))
    print('Total number of events for training:         {}'.format((len(df_sig)+len(df_bkg))))
    print()
    print('Number of signal events for testing:        {}'.format(len(df_sig_test)))
    print('Number of background events for testing:    {}'.format(len(df_bkg_test)))
    print('Total number of events for testing:         {}'.format((len(df_sig_test)+len(df_bkg_test))))
    print()
    print('Total training data:                  {}'.format(len(label_train)))
    print('Total testing data:                   {}'.format(len(label_test)))
    print()
