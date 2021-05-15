# -*- coding: utf-8 -*-
"""
This is the file to plot the SR curve for prey-predator classification 
and concentric circle classification.

@author: harik
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

DATA_NAME = "concentric_circle"

import os
PATH = os.getcwd()
RESULT_PATH = PATH + '/HR-PLOTS/' + DATA_NAME + '/NEUROCHAOS-RESULTS/'


DISCRIMINATION_THRESHOLD = [0.89]
EPSILON = np.arange(0.01, 1.01, 0.01)

FSCORE = np.load(RESULT_PATH + 'h_fscore.npy')

plt.figure(figsize=(10,10))
plt.plot(EPSILON,FSCORE[0,:],'-*k', markersize = 10)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel('Noise intensity', fontsize=30)
plt.ylabel('Average F1-score', fontsize=30)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(RESULT_PATH+"/HR_Chaosnet-"+DATA_NAME+"-SR_plot.jpeg", format='jpeg', dpi=200)
plt.show()
    