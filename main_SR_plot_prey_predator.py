
import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data
from Codes import k_cross_validation
import os

DATA_NAME = "single_variable_classification"
traindata, trainlabel, testdata, testlabel = get_data(DATA_NAME)
FOLD_NO = 5

INITIAL_NEURAL_ACTIVITY = [0.25]
DISCRIMINATION_THRESHOLD = [0.96]
EPSILON = np.arange(0.001, 1.001, 0.001)
FSCORE, Q, B, EPS, EPSILON = k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON, DATA_NAME)


PATH = os.getcwd()
RESULT_PATH = PATH + '/SR-PLOTS/' + DATA_NAME + '/NEUROCHAOS-RESULTS/'
plt.figure(figsize=(10,10))
plt.plot(EPSILON,FSCORE[0,0,:],'-*k', markersize = 10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.xlabel('Noise intensity', fontsize=20)
plt.ylabel('Average F1-score', fontsize=20)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(RESULT_PATH+"/Chaosnet-"+DATA_NAME+"-SR_plot.jpg", format='jpg', dpi=200)
plt.show()
    