# imports from standard python
from __future__ import print_function

# imports from local packages

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
    branchname = 'dnnout_mist'
    branchtype = 'F'

    # read model from training
    n_inputs = len(gl.arg['variables'])
    choose_method = getattr(method, gl.arg['method'])
    model = choose_method(n_inputs=n_inputs)
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
        # open input root file and add branch to the existing tree
        print()
        rootIO.add_branch(ifile, gl.arg['tree'], branchname, branchtype, y)
        print()
        print('DONE!')
        print()

    if len(gl.arg['input']) > 1:
        print('ALL DONE!')
        print()
