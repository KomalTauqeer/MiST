# Script for plotting DNN discriminator output shapes for each node

# Imports from standard python
from __future__ import print_function

# Imports from local packages
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Import six
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Imports from MiST
from MiST import data
from MiST import debugging
from MiST import method
from MiST import plot
from MiST import printout
from MiST import rootIO
from MiST import utilis
from MiST import globaldef as gl
from MiST import plotScript as setup

def plotDiscriminators(true_value,predicted_value,sample):

    opath = gl.arg['mva_path'] + '/'
    true_value_index = np.argmax(true_value, axis = 1)
    predicted_value_index = np.argmax(predicted_value, axis =1)
    class_names = gl.arg['multi_train_plot_labels']
    n_nodes = 8
    
    # Generate one plot per output node
    for node in range(n_nodes):
       
        print("\n PLOTTING OUTPUT NODE '"+ class_names[node] + "'")

        # Initilize list of arrays
        processes = [[], [], [], [], [], [], [], []]

        # Fill list of arrays
        nodePredictions = predicted_value[:,node]
        for event in range(np.add.reduce(np.shape(nodePredictions),0)):
            processes[true_value_index[event]].append(nodePredictions[event])

        processesArrays = [np.asarray(processes[0]), np.asarray(processes[1]), np.asarray(processes[2]), np.asarray(processes[3]), np.asarray(processes[4]), np.asarray(processes[5]), np.asarray(processes[6]), np.asarray(processes[7])]
        # Plot all
        plt.figure()
	colors = ['navy', 'aqua', 'darkturquoise', 'mediumorchid','gold','green','y','blue']
        #plt.hist(processesArrays, bins =18, range=(0.1,1.), color=colors, label =class_names, fill=True, stacked=True)
        #plt.legend()
        #plt.savefig(opath + '/OUT_'+class_names[node] +'.pdf')
        
        # Plot with signal separate
        #reducedProcessesArrays = []
        #reducedColors = []
        #reducedLabels = []
        
        #for process in range(len(processes)):
        #    if process!=node:
        #        reducedProcessesArrays.append(processesArrays[process])
        #        reducedColors.append(colors[process])
        #        reducedLabels.append(class_names[process])

        #plt.hist(reducedProcessesArrays, bins=18, range=(0.1,1.), color=reducedColors, label=reducedLabels, fill=True, stacked=True)
        #plt.hist(processesArrays[node], bins=18, range=(0.1,1.), edgecolor=colors[node], histtype='step',linewidth=1.5, label=class_names[node], fill=False)
        #plt.legend()
        #plt.savefig(opath + '/OUT_'+class_names[node] +'.pdf')


	# Plot with signal in stack
        combinedProcessesArrays = []
        combinedColors = []
        combinedLabels = []
        for process in range(len(processes)):
            combinedProcessesArrays.append(processesArrays[process])
            combinedColors.append(colors[process])
            combinedLabels.append(class_names[process])
        plt.hist(combinedProcessesArrays,bins=20, range=(0.,1.), color=combinedColors, label=combinedLabels, fill=True, stacked=True)
        # scaled signal
        scales = [20,25,20,10,5,15,5,10]
        plt.hist(processesArrays[node], bins=20, range=(0.,1.),weights=np.ones_like(processesArrays[node])*scales[node],edgecolor='red', histtype='step',linewidth=1.5, label=(class_names[node]+' x '+str(scales[node])),fill=False)
        plt.legend()
        plt.xlabel('DNN Output',fontsize=9)
        plt.ylabel('Events/Bin',fontsize=8)
        plt.savefig(opath + '/OUT_'+class_names[node] +'.pdf')
