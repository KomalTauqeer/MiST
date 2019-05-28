# imports from standard python
from __future__ import print_function
import os
import sys
import time
from multiprocessing import Process

# imports from local packages

# imports from pip packages
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import scipy
from tqdm import tqdm
from MiST import globaldef as gl

# imports from MiST


def clear(iplot):
    iplot.clf()
    iplot.cla()
    iplot.close()
    matplotlib.rcParams.update({'font.size': 12})


def inputvars(df_s, df_b, opath, vlist):

    print('--- Creating plots for all input variables...')

    matplotlib.rcParams.update({'font.size': 16})
    plotpath = './' + opath + '/plots'
    if not os.path.isdir(plotpath):
        os.mkdir(plotpath)

    plot_processes = []

    for v in vlist:
        plot_processes.append(Process(target=inputvars_process,
                                      args=(df_s[v], df_b[v], plotpath, v)))

    for p in plot_processes:
        # artificial delay to keep the os happy
        time.sleep(0.05)
        p.start()

    for p in plot_processes:
        p.join()


def inputvars_process(sig, bkg, plotpath, var):

    fig = plt.figure(figsize=(11.69, 8.27), dpi=100)

    bins=np.linspace(min(sig), max(bkg), 30)

    _ = plt.hist(sig,
                 density=True,
                 bins=bins,
                 alpha=0.2,
                 histtype='stepfilled',
                 label='signal',
                 linewidth=2,
                 color='r'
    )

    _ = plt.hist(bkg,
                 density=True,
                 bins=bins,
                 alpha=0.2,
                 histtype='stepfilled',
                 label='background',
                 linewidth=2,
                 color='b'
    )

    plt.xlabel(var.replace('$','\$'))
    plt.legend(loc='best')
    plt.savefig(plotpath + '/plot_' + var + '.pdf')
    clear(plt)


def inputvars_multi(multidata, opath, vlist):

    print('--- Creating plots for all input variables...')

    matplotlib.rcParams.update({'font.size': 16})
    plotpath = './' + opath + '/plots'
    if not os.path.isdir(plotpath):
        os.mkdir(plotpath)

    for key in tqdm(vlist):

        fig = plt.figure(figsize=(11.69, 8.27), dpi=100)

        # -- declare common binning strategy
        bins=np.linspace(min(multidata[0].data()[key]), max(multidata[0].data()[key]), 30)

        for data in multidata:

            _ = plt.hist(data.data()[key],
                         density=True,
                         bins=bins,
                         alpha=0.8,
                         histtype='step',
                         label= data.name,
                         linewidth=2
                         # color='r'
            )

        plt.xlabel(key.replace('$','\$'))
        plt.legend(loc='best')
        plt.savefig(plotpath + '/plot_' + key + '.pdf')

        clear(plt)


def roc(fpr, tpr, auroc, opath, name):

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auroc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(opath + '/roc_' + name + '.pdf')
    clear(plt)


def overtrain(eval_train, label_train, eval_test, label_test, opath):

    n_bins = 30
    #binrange = (-1,1)
    s_color = 'r'
    b_color = 'b'

    s_train = eval_train[label_train==1]
    b_train = eval_train[label_train==0]
    s_test = eval_test[label_test==1]
    b_test = eval_test[label_test==0]

    if min(b_train) < min(b_test):
      binrange_min = min(b_train)
    else:
      binrange_min = min(b_test)

    if max(s_train) > max(s_test):
      binrange_max = max(s_train)
    else:
      binrange_max = max(s_test)

    binrange = (np.asscalar(binrange_min), np.asscalar(binrange_max))

    plt.hist(s_train,
             color=s_color,
             alpha=0.2,
             range=binrange,
             bins=n_bins,
             histtype='stepfilled',
             density=True,
             label='S (training)'
    )

    plt.hist(b_train,
             color=b_color,
             alpha=0.2,
             range=binrange,
             bins=n_bins,
             histtype='stepfilled',
             density=True,
             label='B (training)'
    )


    hist_s_test, bins = np.histogram(s_test,
                                     bins=n_bins,
                                     range=binrange,
                                     density=True
    )

    scale = len(s_test) / sum(hist_s_test)
    err = np.sqrt(hist_s_test * scale) / scale
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    plt.errorbar(center,
                 hist_s_test,
                 color=s_color,
                 yerr=err,
                 fmt='.',
                 ms=2,
                 elinewidth=1,
                 label='S (test)'
    )

    hist_b_test, bins = np.histogram(b_test,
                                     bins=n_bins,
                                     range=binrange,
                                     density=True
    )

    scale = len(b_test) / sum(hist_b_test)
    err = np.sqrt(hist_b_test * scale) / scale

    plt.errorbar(center,
                 hist_b_test,
                 color=b_color,
                 yerr=err,
                 fmt='.',
                 ms=2,
                 elinewidth=1,
                 label='B (test)'
    )

    hist_s_train, bins = np.histogram(s_train,
                                      bins=n_bins,
                                      range=binrange,
                                      density=True
    )

    hist_b_train, bins = np.histogram(b_train,
                                      bins=n_bins,
                                      range=binrange,
                                      density=True
    )

    ks_s = scipy.stats.ks_2samp(np.ravel(s_train), np.ravel(s_test))
    ks_b = scipy.stats.ks_2samp(np.ravel(b_train), np.ravel(b_test))

    plt.xlabel("Classifier output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.title('KS$_{\mathrm{signal}}$ = ' + '{:.3f}'.format(ks_s[1]) + ', KS$_{\mathrm{background}}$ = ' + '{:.3f}'.format(ks_b[1]))

    plt.savefig(opath + '/overtrain.pdf')
    clear(plt)



def training_history(train_dict, opath):

    plt.plot(train_dict['acc'])
    plt.plot(train_dict['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(opath + '/model_accuracy.pdf')
    clear(plt)

    plt.plot(train_dict['loss'])
    plt.plot(train_dict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(opath + '/model_loss.pdf')
    clear(plt)

def plot_confusionMatrix(self, norm_matrix = True, privateWork = False, printROC = False):
        ''' plot confusion matrix '''
        plotCM = plottingScripts.plotConfusionMatrix(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            event_category      = self.categoryLabel,
            plotdir             = self.save_path)

        plotCM.plot(norm_matrix = norm_matrix, privateWork = privateWork, printROC = printROC)


class plotConfusionMatrix:
    def __init__(self, data, prediction_vector, event_classes, event_category, plotdir):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = np.argmax(self.prediction_vector, axis = 1)

        self.event_classes     = event_classes
        self.n_classes         = len(self.event_classes)

        self.event_category    = event_category
        self.plotdir           = plotdir

        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # default settings
        self.ROCScore = None

    def plot(self, norm_matrix = True, privateWork = False, printROC = False):
        if printROC:
            self.ROCScore = roc_auc_score(
                self.data.get_test_labels(), self.prediction_vector)

        # norm confusion matrix if activated
        if norm_matrix:
            new_matrix = np.empty( (self.n_classes, self.n_classes), dtype = np.float64)
            for yit in range(self.n_classes):
                evt_sum = float(sum(self.confusion_matrix[yit,:]))
                for xit in range(self.n_classes):
                    new_matrix[yit,xit] = self.confusion_matrix[yit,xit]/(evt_sum+1e-9)

            self.confusion_matrix = new_matrix


        # initialize Histogram
        cm = setup.setupConfusionMatrix(
            matrix      = self.confusion_matrix.T,
            ncls        = self.n_classes,
            xtitle      = "predicted class",
            ytitle      = "true class",
            binlabel    = self.event_classes)

        canvas = setup.drawConfusionMatrixOnCanvas(cm, "confusion matrix", self.event_category, self.ROCScore, privateWork = privateWork)
        setup.saveCanvas(canvas, self.plotdir+"/confusionMatrix.pdf")

