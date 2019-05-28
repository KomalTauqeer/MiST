# imports from standard python
from __future__ import print_function
import os
import sys
# import re
import shutil
# import six
import time
import hashlib
import math

# imports from local packages

# imports from pip packages
from sklearn.metrics import roc_curve, roc_auc_score

# imports from MiST
from MiST import globaldef as gl
from MiST import plot

pi = math.pi

def arg_consistency_check():

    n_err = 0
    ncores_total = 4
    bad_paths = ['MiST','docs','config']

    if gl.arg['gpucores'] > ncores_total:
        print('ERROR: more GPU cores requested than available!')
        n_err += 1

    if gl.arg['gpucores'] < 1:
        print('ERROR: please request a valid number of GPU cores (>=1)!')
        n_err += 1

    if gl.arg['type'] == 'train' or gl.arg['type'] == 'dev':

        tmp_list = gl.arg['signal']
        tmp_list = [gl.arg['signal']] if isinstance(tmp_list, str) else tmp_list
        for i in tmp_list:
            if not os.path.isfile(i):
                print('ERROR: signal file for training not found: ' + i)
                n_err += 1
        tmp_list = gl.arg['background']
        tmp_list = [gl.arg['background']] if isinstance(tmp_list, str) else tmp_list
        for i in tmp_list:
            if not os.path.isfile(i):
                print('ERROR: background file for training not found: ' + i)
                n_err += 1
        tmp_list = gl.arg['signal_test']
        tmp_list = [gl.arg['signal_test']] if isinstance(tmp_list, str) else tmp_list
        for i in tmp_list:
            if not os.path.isfile(i):
                print('ERROR: signal test file for training not found: ' + i)
                n_err += 1
        tmp_list = gl.arg['background_test']
        tmp_list = [gl.arg['background_test']] if isinstance(tmp_list, str) else tmp_list
        for i in tmp_list:
            if not os.path.isfile(i):
                print('ERROR: background test file for training not found: ' + i)
                n_err += 1

    if gl.arg['type'] == 'eval':

        for ifile in gl.arg['input']:
            if not os.path.isfile(ifile):
                print('ERROR: input file for eval not found: ' + gl.arg['input'])
                n_err += 1

    # if not hasattr(method, gl.arg['method']):
    #     print('ERROR: method not found: ' + gl.arg['method'])
    #     n_err += 1

    for bad in bad_paths:
        if bad == gl.arg['mva_path']:
            print('ERROR: protected dir name!: ', bad)
            n_err += 1

    if not 0 < gl.arg['validationsize'] < 1:
        print('ERROR: validation fraction needs to between 0 and 1')
        n_err += 1

    if n_err > 0:
        print('EXIT: ' + str(n_err) + ' error(s)')
        sys.exit(1)


def rmdir(path):
    try:
        shutil.rmtree(path)
    except IOError:
        print('no permission to remove ' + path + '!')
        sys.exit(1)


def mkdir(path):
    try:
        os.mkdir(path)
    except IOError:
        print('no permission to create ' + path + '!')
        sys.exit(1)


def training_path(path):
    if os.path.isdir(path):
        sys.stdout.write('WARNING: Output path already exists, Ctrl+C to abort overwriting in ')
        for i in reversed(range(1,6)):
            sys.stdout.write(str(i))
            sys.stdout.flush()
            for j in range(3):
                time.sleep(0.333)
                sys.stdout.write('.')
                sys.stdout.flush()
        rmdir(path)
    mkdir(path)
    mkdir(path + '/model')
    print()


def conv_str_to_list(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default, split on a comma and strip whitespaces."""
    return [ chunk.strip(chars) for chunk in option.split(sep) ]


def resolve_config_string(textstring, replace_comma=False):
    textstring = textstring.replace("\n"," ")
    textstring = textstring.replace("\t"," ")
    textstring = textstring.replace("\r"," ")
    if replace_comma:
        textstring = textstring.replace(","," ")
    textstring = ' '.join(textstring.split())
    return textstring


def get_hash():
    hstring = ''.join(gl.arg['variables'])+gl.arg['method']
    hash_object = hashlib.sha1(hstring.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def write_hash():

    hstring = get_hash()

    hfile = gl.arg['mva_path'] + '/model/' + gl.hfilename

    try:
        text_file = open(hfile, "w")
        text_file.write(hstring)
        text_file.close()
    except IOError:
        print('something went wrong while trying to write the hash')
        print(hstring)
        print(hfile)


def read_hash():

    hfile = gl.arg['mva_path'] + '/model/' + gl.hfilename

    try:
        with open(hfile, 'r') as myfile:
            data = myfile.read().replace('\n', '')
        return data
    except IOError:
        print('ERROR: could not open hash file %s' % hfile)
        sys.exit(1)


def comp_hash():

    hstring1 = get_hash()
    hstring2 = read_hash()

    if hstring1 != hstring2:

        print('ERROR: variables and/or method not compatible with the training!')
        print('training:')
        print(hstring1)
        print('eval:')
        print(hstring2)

        sys.exit(1)


def add_tb_link():
    cmd_fname = gl.arg['mva_path'] + '/start_tb.sh'
    cmd_str = 'tensorboard --logdir=./TensorBoard'
    cmd_file = open(cmd_fname, "w")
    cmd_file.write(cmd_str)
    cmd_file.close()


def roc(labels, values, opath, name):
    # ROC curve from sklearn
    fpr, tpr, thresholds = roc_curve(labels,
                                     values,
                                     pos_label=None,
                                     sample_weight=None,
                                     drop_intermediate=True)

    # print(fpr)
    # print(tpr)
    # print(thresholds)

    print('ROC ' + name + ':')
    auroc = roc_auc_score(labels, values)
    print(auroc)

    plot.roc(fpr, tpr, auroc, opath, name)


def convert_list_to_dict(ilist):

    output_dict = {}

    for i in ilist:
        if 'true' in i[1] or 'True' in i[1]:
            output_dict[i[0]] = True
        elif 'false' in i[1] or 'False' in i[1]:
            output_dict[i[0]] = False
        elif '.' in i[1] or '1e' in i[1]:
            output_dict[i[0]] = float(i[1])
        elif 'epsilon' in i[0] and 'None' in i[1]:
            output_dict[i[0]] = None
        elif any(char.isdigit() for char in i[1]):
            output_dict[i[0]] = int(i[1])
        else:
            print('### unknown type: ' + str(i))

    return output_dict


def deltaPhi(phi1, phi2):

    result = phi1 - phi2

    while result > pi:
        result -= 2*pi

    while result <= -pi:
        result += 2*pi

    return result


def deltaR(eta1, phi1, eta2, phi2):

    deta = eta1 - eta2

    dphi = deltaPhi(phi1, phi2)

    return math.sqrt(deta*deta + dphi*dphi)
