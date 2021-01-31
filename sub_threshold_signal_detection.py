"""
Author: Harikrishnan N B
Email: harikrishnannb07@gmail.com

Sub threshold Signal Detection
------------------------------
Computes the cross correlation coefficient between the output
of ChaosNet (normalized firing time) and input signal for
varying noise intensties. 
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
import os
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.svm import LinearSVC

import ChaosFEX.feature_extractor as CFX
# Input periodic signal
A = 0.2
t = np.arange(0, 1, 0.001) # time
threshold = 0.5 # Global threshold
## Sub threshold signal
signal = A * (np.sin(2*np.pi*5*t) + 1)/2  # Input Sub Threshold Signal
input_data = signal.reshape(len(signal),1)

INA = 0.35 # Initial Neural Activity
EPSILON = np.arange(0.001, 1.001, 0.001)
#EPSILON_1 = 0.05 # Epsilon
DT = 0.65 # Discrimination Threshold

ROW = -1
CROSS_CORR_ZERO_LAG = np.zeros((len(EPSILON), 1))
compute_statistics = 2 # [0-firing rate, 1 - Energy, 2 - Firinig Time and 3 - Entropy]


feature_name =['firing rate', 'energy', 'firing time', 'entropy']
for EPSILON_1 in EPSILON:
    ROW = ROW+1
    feature = CFX.transform(input_data, INA, 10000, EPSILON_1, DT)[:, compute_statistics]
    feature_quantized = feature.copy()
    if compute_statistics == 1 or compute_statistics == 2:
       
        feature=(feature - np.min(feature))/(np.max(feature) - np.min(feature)+0.0001)
        
    if np.std(feature) != 0:
        CROSS_CORR_ZERO_LAG[ROW, 0] = np.corrcoef(signal, feature)[0,1]
   
        
    

plt.figure(figsize=(10,10))
plt.plot(EPSILON, CROSS_CORR_ZERO_LAG,'-*k', markersize = 6)
# plt.axhline(y=threshold, color='b', linestyle='--', alpha=0.6, label = 'threshold')
# plt.plot(t, feature,'-or', markersize = 6, label = feature_name[compute_statistics])


#plt.plot(t, (firing_rate-np.min(firing_rate))/(np.max(firing_rate)-np.min(firing_rate)),'-or', markersize = 6, label = 'normalized firing time')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.xlabel('Noise intensity', fontsize=20)
plt.ylabel('Cross Correlation Coefficient', fontsize=20)
#plt.ylim(0, 1)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("cross-correlation-coefficient-plot-signal-detection.jpg", format='jpg', dpi=200)

plt.show()
MAX_CROSS_CORR_LIST = []
OPTIMUM_NOISE_LIST = []
MAX_CROSS_CORR = np.max(CROSS_CORR_ZERO_LAG)
for row in range(0, len(EPSILON)):
    if CROSS_CORR_ZERO_LAG[row] == MAX_CROSS_CORR:
        MAX_CROSS_CORR_LIST.append(MAX_CROSS_CORR)
        OPTIMUM_NOISE_LIST.append(EPSILON[row])
        
        
print("Maximum Cross correlation = ", MAX_CROSS_CORR_LIST, "at Noise intensity = ", OPTIMUM_NOISE_LIST)