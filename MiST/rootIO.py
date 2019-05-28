# imports from standard python
from __future__ import print_function
import sys
from array import array
import glob
import shutil
import time
from multiprocessing import Process, Manager


# import from local packages
from ROOT import TFile, TTree, TH1F

# imports from pip packages
from tqdm import tqdm
import pandas as pd
import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_pandas import read_root, to_root
from root_numpy import root2array, tree2array, array2tree, array2root, root2rec

# imports from MiST


def checkfile(filename, treename):
    # probably doesn't really check
    # nope, it does not
    # (py)root sucks
    try:
        ifile = TFile(filename, 'READ')
    except:
        print('ERROR: cannot open ROOT file: %s', filename)
        sys.exit(1)
    checktree(filename, treename, ifile)


def checktree(filename, treename, ifile):
    try:
        itree = ifile.Get(treename)
        # itree.Print()
    except ReferenceError:
        print('ERROR: cannot open ROOT tree %s in ROOT file %s', (treename, filename))
        sys.exit(1)


def getcrosseff(filename,tree):

    ifile = TFile("%s" % filename, 'READ')

    itree = ifile.Get(tree)

    tmp = 1.0
    for i in itree.GetUserInfo():
        if str(i).startswith("efficiency") or str(i).startswith("xsection"):
            tmp = tmp * float(str(i).split(":")[1])

    ifile.Close()

    return tmp


def root2df(files_path, tree_name, **kwargs):
        """
        Args:
        -----
        files_path: a string like './data/*.root', for example
        tree_name: a string like 'Collection_Tree' corresponding to the name of the folder inside the root file that we want to open
        kwargs: arguments taken by root2array, such as branches to consider, start, stop, step, etc
        Returns:
        --------
        output_panda: a pandas dataframe like allbkg_df in which all the info from the root file will be stored

        Note:
        -----
        if you are working with .root files that contain different branches, you might have to mask your data in that case, return pd.DataFrame(ss.data)
        """
        # -- create list of .root files to process
        files = glob.glob(files_path)

        # -- process ntuples into rec arrays
        #ss = stack_arrays([root2rec(fpath, tree_name, **kwargs) for fpath in files]) <--- deprecated
        ss = stack_arrays([root2array(fpath, tree_name, **kwargs).view(np.recarray) for fpath in files])

        try:
            return pd.DataFrame(ss)
        except Exception:
            return pd.DataFrame(ss.data)



'''
def load_df(file, tree, vlist):


    if file == 'None':
        print('No input file!')
        sys.exit(1)

    vlist_noexpand = []
    v_to_flatten = []
    for v in vlist:
        if ('[' in v) and (']' in v):
            vlist_noexpand.append('noexpand:' + v)
            v_to_flatten.append(v)
        else:
            vlist_noexpand.append(v)


    df = read_root(paths=file,
                   key=tree,
                   columns=vlist_noexpand,
                   flatten=v_to_flatten,
                   ignore=None,
                   chunksize=None,
                   where=None
    )


    return df
'''


def load_df(ifile, tree, list_noexpand, to_flatten):

    df = read_root(paths=ifile,
                   key=tree,
                   columns=list_noexpand,
                   flatten=to_flatten,
                   ignore=None,
                   chunksize=None,
                   where=None
    )

    return df


def load_weights(ifile, tree, weights):

    if ifile == 'None':
        print('No input file!')
        sys.exit(1)

    vlist_noexpand = []
    v_to_flatten = []
    for v in weights:
        if ('[' in v) and (']' in v):
            vlist_noexpand.append('noexpand:' + v)
            v_to_flatten.append(v)
        else:
            vlist_noexpand.append(v)

    df = read_root(paths=ifile,
                   key=tree,
                   columns=vlist_noexpand,
                   flatten=v_to_flatten,
                   ignore=None,
                   chunksize=None,
                   where=None
    )

    return df


def add_branch(filename, treename, branchname, branchtype, data):

    # open input file and tree
    ifile = TFile(filename,'READ')
    itree = ifile.Get(treename)

    # create output file
    ofile = TFile(filename+'.mist','RECREATE')

    # clone tree, FIX: hardcoded
    ofile.mkdir('utm')
    ofile.cd('utm')

    # set branch inactive in itree if it already exists
    if itree.FindBranch(branchname):
        itree.SetBranchStatus(branchname,0)

    # clone itree
    print('--- Cloning input file ...')
    otree = itree.CloneTree()
    otree.Write()

    # close input file
    ifile.Close()

    # make new variable and add it as a branch to the tree
    y_helper = array(branchtype.lower(),[0])
    branch = otree.Branch(branchname, y_helper, branchname + '/' + branchtype)

    # get number of entries and check if size matches the data
    n_entries = otree.GetEntries()
    print('entries are here')
    print(n_entries)
    if n_entries != data.size:
        print('mismatch in input tree entries and new branch entries!')

    # fill the branch
    print('--- Adding branch %s in %s:%s ...' %(branchname, filename, treename))
    for i in tqdm(xrange(n_entries)):
        otree.GetEntry(i)
        y_helper[0] = data[i]
        branch.Fill()

    # write new branch to the tree and close the file
    ofile.Write("",TFile.kOverwrite)
    ofile.Close()

    # overwrite old file
    print('--- Overwrite original file ...')
    shutil.move(filename + '.mist', filename)


def save_training_inputs(data_train, label_train, opath, vlist, mode):

    dirname = 'inputs_' + mode
    fullname = opath + 'training.root'

    ofile = TFile(fullname, 'update')
    ofile.mkdir(dirname)
    ofile.cd(dirname)

    n_bins = 100
    # determine range for all variables
    x_min = np.min(data_train, 0)
    x_max = np.max(data_train, 0)

    print('--- Creating histograms of input variables - %s' % mode)

    manager = Manager()
    h_output = manager.list()
    h_processes = []

    n_v = 0
    for v in vlist:
        h_processes.append(Process(target = save_training_inputs_process,
                                   args = (data_train, label_train, n_v, v, n_bins, x_min, x_max, h_output)))
        n_v += 1


    for p in h_processes:
        time.sleep(0.05)
        p.start()

    for p in h_processes:
        p.join()

    for h in h_output:
        h.Write()

    ofile.Close()


def save_training_inputs_process(data, label, n, var, nbins, xmin, xmax, h_output):

    h_s = TH1F(var+'_s', var+'_s', nbins, xmin[n]-1, xmax[n]+1)
    h_b = TH1F(var+'_b', var+'_b', nbins, xmin[n]-1, xmax[n]+1)

    for x, y in zip(data, label):
        if y == 0:
            h_b.Fill(x[n])
        elif y == 1:
            h_s.Fill(x[n])
        else:
            print('this should not happen..')

    h_output.append(h_s)
    h_output.append(h_b)


def save_training_reults(**kwargs):

    label_train = kwargs['label_train']
    eval_train = kwargs['eval_train']
    label_test = kwargs['label_test']
    eval_test = kwargs['eval_test']
    opath = kwargs['opath']

    # dirname = 'inputs_' + mode
    fullname = opath + 'training.root'

    ofile = TFile(fullname, 'update')
    # ofile.mkdir(dirname)
    # ofile.cd(dirname)

    h_s_train = TH1F('signal_training', 'signal_training', 50, 0, 1)
    h_s_test = TH1F('signal_test', 'signal_test', 50, 0, 1)
    h_b_train = TH1F('background_training', 'background_training', 50, 0, 1)
    h_b_test = TH1F('background_test', 'background_test', 50, 0, 1)

    print('--- Saving shapes for training data...')
    for value, label in tqdm(zip(eval_train, label_train)):
        if label == 0:
            h_b_train.Fill(value)
        elif label == 1:
            h_s_train.Fill(value)
        else:
            print('this should not happen..')


    print('--- Saving shapes for test data...')
    for value, label in tqdm(zip(eval_test, label_test)):
        if label == 0:
            h_b_test.Fill(value)
        elif label == 1:
            h_s_test.Fill(value)
        else:
            print('this should not happen..')

    ofile.Write()
    ofile.Close()


def save_multi_training_reults(**kwargs):

    label_train = kwargs['label_train']
    eval_train = kwargs['eval_train']
    label_test = kwargs['label_test']
    eval_test = kwargs['eval_test']
    n_outputs = kwargs['n_outputs']
    opath = kwargs['opath']

    fullname = opath + 'training.root'

    ofile = TFile(fullname, 'update')

    nbinsx = 50
    xmin = 0
    xmax = 1

    l_outputs = range(n_outputs)

    h_train = [0] * n_outputs
    h_test = [0] * n_outputs

    for i in l_outputs:
        h_train[i] = [0] * n_outputs
        h_test[i] = [0] * n_outputs

    # i is the correct class, j is the node output
    for i in l_outputs:
        for j in l_outputs:

            tmp_class_xaxis = 'class {}, output class {}'.format(i,j)
            tmp_class_histo = 'class_{}_output_{}'.format(i,j)

            h_train[i][j] = TH1F('train_' + tmp_class_histo, 'train_' + tmp_class_histo, nbinsx, xmin, xmax)
            h_test[i][j] =  TH1F('test_' + tmp_class_histo, 'test_' + tmp_class_histo, nbinsx, xmin, xmax)

            h_train[i][j].GetXaxis().SetTitle(tmp_class_xaxis)
            h_test[i][j].GetXaxis().SetTitle(tmp_class_xaxis)

    print('--- Saving shapes for training data...')

    # loop over all events in
    for evt in tqdm(range(label_train.shape[0])):
        # loop over possible classes
        for i in l_outputs:
            # find true class
            if label_train[evt][i] == 1:
                # fill all output values for that one true class
                for j in l_outputs:
                    h_train[i][j].Fill(eval_train[evt][j])

    print('--- Saving shapes for test data...')

    for evt in tqdm(range(label_test.shape[0])):
        for i in l_outputs:
            if label_test[evt][i] == 1:
                for j in l_outputs:
                    h_test[i][j].Fill(eval_test[evt][j])

    ofile.Write()
    ofile.Close()
