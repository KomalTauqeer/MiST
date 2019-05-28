# imports from standard python
from __future__ import print_function
import sys
import itertools

# imports from local packages

# imports from pip packages
import pandas as pd

# imports from MiST


"""module to provide MC truth matching for jet-parton assignment"""


class MCTruthMatching:

    def __init__(self):

        self.matchingtype = ''
        self.data = None
        self.truth = None

        self.hyp_correct_train = pd.DataFrame()
        self.hyp_wrong_train = pd.DataFrame()
        self.hyp_correct_test = pd.DataFrame()
        self.hyp_wrong_test = pd.DataFrame()

        self.hyp_train = pd.DataFrame()
        self.hyp_test = pd.DataFrame()

        print('--- MC truth matching enabled')

    def add(self, data_handler_reco):

        self.data = data_handler_reco.dataobj
        self.truth = data_handler_reco.truthobj

    def apply_matching(self, matchingtype):

        print('--- Matching will be applied for %s hypothesis' % matchingtype)

        self.matchingtype = matchingtype

        if self.matchingtype is 'thq':

            self.test()
            #self.matching_thq()

        elif self.matchingtype is 'thw':

            self.matching_thw()

        elif self.matchingtype is 'ttbar' or 'tt':

            self.matching_ttbar()

        else:

            print('ERROR: matching type unknown: ' + matchingtype)
            sys.exit(1)

    def matching_thq(self):

        pass

    def matching_thw(self):

        pass

    def matching_ttbar(self):

        pass

    def data_train_test(self):

        return self.hyp_train, self.hyp_test

    def test(self):

        # stuff to be matched to jets
        partons_list = ['btop', 'hbb1', 'hbb2', 'lq']

        # create permutations to iterate over
        perm = list(itertools.permutations(partons_list))
        print(perm)

        ievt = 0

        # iterate over all events, two options, both kinda suck:
        # iterrows: does not preserve dtype
        # for idx, evt in idata.iterrows():
        # itertuples: modifies iterated object on the fly
        # for evt in idata.itertuples(index=True, name='Pandas'):

        # need to access variables and truth variables with same index, so better use indexing
        n_evt = len(self.data.index)

        if n_evt != len(self.truth.index):
            print('ERROR: mismatch of entries in variables and truth variables: {} vs {}'.format(n_evt, len(self.truth.index)))
            sys.exit(1)

        n_evt_range = range(n_evt)

        pd.set_option('max_colwidth', 800)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        for i in n_evt_range:

            # get variables for this specific event
            i_vars = self.data.iloc[i]
            i_truth = self.truth.iloc[i]

            print('### now ievt ' + str(i))

            print('### kinematics: ')
            print(i_vars)
            print()

            print('### truth: ')
            print(i_truth)
            print()

            # get number of jets in the event
            njets_total = len(i_vars[0])
            njets_total_range = range(njets_total)

            print('event has {} jets in total'.format(njets_total))
            print(njets_total_range)

            # list of jet pt and eta for selection, row index still is i...
            #j_pt = i_vars.at[i,'j_pt']
            #j_eta = i_vars.at[i,'j_eta']
            j_pt = i_vars[ 'j_pt']
            j_eta = i_vars['j_eta']

            print(j_pt)
            print(j_eta)

            # prepare list of jet indices that contain 'good' jets
            njets_selected = 0
            list_jets_selected = []

            # loop over all jets
            for j in njets_total_range:

                # check if jet passes the selection and store index
                if (j_pt[j] > 30 and abs(j_eta[j]) < 2.4) or j_pt[j] > 40:
                    njets_selected += 1
                    list_jets_selected.append(j)

            print('found {} jets for matching'.format(njets_selected))
            print(list_jets_selected)





            ievt = ievt + 1
            if ievt > 6:
                break

        #print(self.data)
        #print(self.truth)

        print('done matching test')
