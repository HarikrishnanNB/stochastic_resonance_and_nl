# -*- coding: utf-8 -*-
"""
This is the code corresponding to five fold crossvalidation 
using Hindmarsh Rose Neuronal model in NL architecture. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)


from chaos_codes import chaos_transform, chaosnet, k_cross_validation

from load_data import get_data
from hindmarsh_rose import hr_trajectory
# =============================================================================
# Parameters of Hindmarsh Rose Model in Chaotic Regime
# Refer the following paper for the parameters in chaotic regim.
# Holden, Arun V., and Yin-Shui Fan. "From simple to simple bursting oscillatory behaviour via chaos in the Rose-Hindmarsh model for neuronal activity." Chaos, Solitons & Fractals 2.3 (1992): 221-236.
# =============================================================================


X0 = [0, 0, 0] 
current = 3.28

t = 2000
dt = 0.01
a = 1
b = 3
c = 1
d = 5
r = 0.0039
s = 4
x1 = -8/5
tvec = np.arange(0, t, dt)
x, y, z = hr_trajectory(X0, t, current, dt, r, x1, a, b, c, d, s)     


DATA_NAME = "single_variable_classification"
traindata, trainlabel, testdata, testlabel = get_data(DATA_NAME)


normalized_trajectory = (x - np.min(x))/(np.max(x) - np.min(x))
timeseries = np.reshape(normalized_trajectory, (len(x), 1))

FOLD_NO = 5


DISCRIMINATION_THRESHOLD = [0.89]
EPSILON = np.arange(0.01, 1.01, 0.01)
FSCORE, B, EPS, EPSILON = k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, DISCRIMINATION_THRESHOLD, EPSILON, timeseries, DATA_NAME)

DATA_NAME = "concentric_circle"
import os
PATH = os.getcwd()
RESULT_PATH = PATH + '/HR-PLOTS/' + DATA_NAME + '/NEUROCHAOS-RESULTS/'

# EPSILON = np.arange(0.01, 1.01, 0.01)
# FSCORE = np.load(RESULT_PATH + 'h_fscore.npy')

plt.figure(figsize=(10,10))
plt.plot(EPSILON,FSCORE[0,:],'-*k', markersize = 10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.xlabel('Noise intensity', fontsize=20)
plt.ylabel('Average F1-score', fontsize=20)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(RESULT_PATH+"/HR_Chaosnet-"+DATA_NAME+"-SR_plot.jpeg", format='jpeg', dpi=200)
plt.show()
    