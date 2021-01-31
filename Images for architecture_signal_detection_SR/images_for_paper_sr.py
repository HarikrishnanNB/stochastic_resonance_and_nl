# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com
Images used to explain the architecture of SR and NL (Refer Paper)
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
signal = A * (np.sin(2*np.pi*5*t) + 1)/2  # Input Sub Threshold Signal
input_data = signal.reshape(len(signal),1)

INA = 0.35 # Initial Neural Activity

 # Epsilon
DT = 0.65 # Discrimination Threshold

NOISE_TYPE = "high-noise" #NOISE_TYPE can take the following
# "high-noise" OR "low-noise"  OR "optimum"


if NOISE_TYPE == "low-noise":
    EPSILON_1 = 0.001
elif NOISE_TYPE == "high-noise":
    EPSILON_1 = 0.95
elif NOISE_TYPE == "optimum":
    EPSILON_1 = 0.033
compute_statistics = 2 # [0-firing rate, 1 - Energy, 2 - Firinig Time and 3 - Entropy]


feature_name =['firing rate', 'energy', 'firing time', 'entropy']

feature = CFX.transform(input_data, INA, 10000, EPSILON_1, DT)[:, compute_statistics]
length = len(signal)
NOISE_LEVEL = []    
for row in range(0, length):
    NOISE_LEVEL.append(np.random.uniform((signal[row]-EPSILON_1), (signal[row]+ EPSILON_1)))
   



if compute_statistics == 1 or compute_statistics == 2:
    feature=(feature - np.min(feature))/(np.max(feature) - np.min(feature) + 0.0001)
    # feature_quantized = feature.copy()
    # feature_quantized[feature > threshold] = 1
    # feature_quantized[feature <= threshold] = 0
plt.figure(figsize=(10,10))
plt.plot(t, signal,'-k', markersize = 6, linewidth=5.0,label = r'$x(t)$')
#plt.plot(t, feature_quantized,'--k', linewidth=3.0, label = r'$\^y(t)$')
plt.axhline(y=threshold, color='b',linewidth=5.0, linestyle='--', alpha=1.0, label = r'$x_{th}$')
# plt.plot(t, feature,'-or', markersize = 6, label = r'$y(t)$')


#plt.plot(t, (firing_rate-np.min(firing_rate))/(np.max(firing_rate)-np.min(firing_rate)),'-or', markersize = 6, label = 'normalized firing time')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid(True)
plt.xlabel('t', fontsize=30)
# plt.ylabel('Signal', fontsize=20)
plt.ylim(-1, 1)
plt.legend(fontsize=30)
plt.tight_layout()
plt.savefig("Paper-images/"+"input-threshold.jpg", format='jpg', dpi=200)
# plt.savefig("Paper-images/"+NOISE_TYPE+"-output-alone.jpg", format='jpg', dpi=200)

plt.show()


plt.figure(figsize=(10,10))
#plt.plot(t, signal,'-k', markersize = 6, linewidth=2.0,label = r'$x(t)$')
#plt.plot(t, feature_quantized,'--k', linewidth=3.0, label = r'$\^y(t)$')
#plt.axhline(y=threshold, color='b', linestyle='--', alpha=0.6, label = r'$x_{th}$')
plt.plot(t, NOISE_LEVEL,'-k',linewidth=4.0, markersize = 6)
#plt.plot(t, signal - EPSILON_1,'--r', markersize = 6, alpha = 0.8)
#plt.plot(t, signal + EPSILON_1,'--r', markersize = 6, alpha = 0.8)
#plt.plot(t, (firing_rate-np.min(firing_rate))/(np.max(firing_rate)-np.min(firing_rate)),'-or', markersize = 6, label = 'normalized firing time')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid(True)
plt.xlabel('t', fontsize=30)
# plt.ylabel('Signal', fontsize=20)
plt.ylim(-1.15, 2.15)
#plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("Paper-images/"+NOISE_TYPE+"-input_noise.jpg", format='jpg', dpi=200)

plt.show()

# TIME_SERIES = np.vstack((input_data.T, feature_quantized.reshape(1, len(feature_quantized))))
if np.std(feature) == 0:
        print("Cross correlation coefficient = ", 0)
else:
    print("Cross correlation coefficient = ", np.corrcoef(signal, feature)[0,1])     
   

    
