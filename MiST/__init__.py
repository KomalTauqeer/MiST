#!/usr/bin/env python

# imports from standard python
from __future__ import print_function
import os
import sys

# imports from local packages

# imports from pip packages
import keras
import six

# imports from MiST
from MiST import eval
from MiST import train
from MiST import multi_train
from MiST import utilis
from MiST import reco_train
from MiST import settings
from MiST import globaldef as gl


def init():

    # fancy logo
    logo()

    # parse final settings from argparse+configparse
    arg = settings.options()

    # make some widely used arguments available as global varaibles
    gl.arg = arg

    # check if everything is reasonable
    utilis.arg_consistency_check()

    # set tf log level to disable most warnings: TF_CPP_MIN_LOG_LEVEL=2
    if not arg['verbose']:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # start tensorflow in advance so we can set a few options
    settings.init_tf(arg['gpucores'], arg['verbose'])

    print()
    print('-' * gl.screenwidth)
    # call the right function to go on
    if arg['type'] == 'train':
        print('--- Running in training mode')
        print('-' * gl.screenwidth)
        train.init()
    elif arg['type'] == 'eval':
        print('--- Running in evaluation mode')
        print('-' * gl.screenwidth)
        eval.init()
    elif arg['type'] == 'multi_train':
        print('--- Running in multi training mode')
        print('-' * gl.screenwidth)
        multi_train.init()
    elif arg['type'] == 'reco_train':
        print('--- Running in multi training mode')
        print('-' * gl.screenwidth)
        reco_train.init(arg)

    else:
        print('### init type exception')


def logo():

    logocolorstart = 242

    try:
        print()
        logofile = open('MiST/logo/logo.txt', 'r')
        logocolor = logocolorstart
        for line in logofile:
            print('\x1b[38;5;'+str(logocolor)+'m'+line,end='')
            logocolor = logocolor + 1
        print('\x1b[0m')
    except:
        print()

    print()
