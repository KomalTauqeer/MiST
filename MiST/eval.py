# imports from standard python
from __future__ import print_function

# imports from local packages
import numpy as np
# imports from pip packages

# imports from MiST
from MiST import data
from MiST import rootIO
from MiST import utilis
from MiST import globaldef as gl
from MiST import method


def init():

    # check if eval is compatible with training
    utilis.comp_hash()

    # some definitions
    branchname =[ 'dnnout_mist0','dnnout_mist1','dnnout_mist2','dnnout_mist3','dnnout_mist4','dnnout_mist5','dnnout_mist6','dnnout_mist7']
    branchtype = 'F'

    # read model from training
    n_inputs = len(gl.arg['variables'])
    tmp_n_outputs = np.shape(gl.arg['input'])
    n_outputs = np.add.reduce(tmp_n_outputs,0)

    choose_method = getattr(method, gl.arg['method'])
    model = choose_method(n_inputs=n_inputs,n_outputs=n_outputs)
    model.load()

    for ifile in gl.arg['input']:
        print('--- Now file %s' % ifile)

        # open the input file and get the variables
        idata = data.DataHandlerEvalBinary()
        idata.add(ifile, gl.arg['tree'])

        # convert to matrix for easier use
        data_input = idata.data().values

        # transform input data in the same way as the training data
        data_input = data.apply_transform(data_input, gl.arg['mva_path'])

        # evaluate for the input data
        print('--- Evaluating model...')
        y = model.eval(data_input)
        print(y)
        print(np.shape(y)) 
        
        # open input root file and add branch to the existing tree
        print()
        rootIO.add_branch_multi(ifile, gl.arg['tree'],n_outputs, branchname, branchtype, y)
        print()
        print('DONE!')
        print()

    if len(gl.arg['input']) > 1:
        print('ALL DONE!')
        print()
