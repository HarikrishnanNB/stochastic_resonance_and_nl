
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com

The below code generates plot for low noise, high noise and zero noise 
for a particular stimulus.

"""


import numpy as np
import matplotlib.pyplot as plt
import ChaosFEX.feature_extractor as CFX
import ChaosFEX.chaotic_sampler as CHAOS


init_cond = 0.21
threshold = 0.499
length = 200

NOISE_TYPE = "high noise" # Type of noise

X_TRAIN = np.array([[0.81]])# Stimulus

if NOISE_TYPE == "low noise":
    EPSILON = 0.01
elif NOISE_TYPE == "high noise":
    EPSILON = 0.15
elif NOISE_TYPE == "zero noise":
    EPSILON = 0


sub_trajectory = np.zeros((length, 1))
trajectory = CHAOS.compute_trajectory(init_cond, threshold, length, validate=False)


if NOISE_TYPE == 'zero noise':
    
    plt.figure(figsize=(12,12))
    plt.axhline(y=X_TRAIN[0,0], color='r', linestyle='-', label='stimulus')
    # plt.axhline(y=X_TRAIN[0,0]- EPSILON, color='b', linestyle='--', alpha=0.6)
    # plt.axhline(y=X_TRAIN[0,0]+ EPSILON, color='b', linestyle='--', alpha=0.6)
    firingtime = np.arange(0, length, 1)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel('Firing Time (ms)', fontsize=40)
    plt.ylabel('Neural Response', fontsize=40)
    plt.ylim(0,1)
    plt.xlim(-0.1, length+20)
    plt.plot(firingtime, trajectory, '-*k', alpha=0.9, linewidth=1.5, label='neural response')
    # plt.plot(firingtime, NOISE_LEVEL, 'b', alpha=0.9, label=NOISE_TYPE)
    plt.plot([205,210, 215],[0.5, 0.5, 0.5], 'ok', markersize=8)
    plt.grid()
    plt.legend(fontsize=35)
    plt.tight_layout()
    plt.savefig(NOISE_TYPE+".jpeg", format='jpeg', dpi=300)
    plt.show()
else:
    FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, init_cond, length, EPSILON, threshold)
    A = (np.abs((X_TRAIN[0,0]) - trajectory) < EPSILON)
    
    sub_trajectory[0:np.int(A.tolist().index(True)+1),0] = trajectory[0:np.int(A.tolist().index(True)+1)]
    
    NOISE_LEVEL = []    
    for rand_length in range(0, length):
        NOISE_LEVEL.append(np.random.uniform((X_TRAIN[0,0]-EPSILON), (X_TRAIN[0,0]+ EPSILON)))
    
    plt.figure(figsize=(12,12))
    plt.axhline(y=X_TRAIN[0,0], color='r', linestyle='-', label='stimulus')
    plt.axhline(y=X_TRAIN[0,0]- EPSILON, color='b', linestyle='--', alpha=0.6)
    plt.axhline(y=X_TRAIN[0,0]+ EPSILON, color='b', linestyle='--', alpha=0.6)
    firingtime = np.arange(0, length, 1)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel('Firing Time (ms)', fontsize=40)
    plt.ylabel('Neural Response', fontsize=40)
    plt.ylim(0,1)
    plt.xlim(-0.1, length)
    plt.plot(firingtime, sub_trajectory, '-*k', alpha=0.9,linewidth=1.5, label='neural response')
    plt.plot(firingtime, NOISE_LEVEL, 'b', alpha=0.9, label='stimulus + noise')
    
    # plt.annotate('Brackmard minimum',
    # ha = 'right', va = 'bottom',
    # xytext = (np.int(A.tolist().index(True)), 0.6),xy = (np.int(A.tolist().index(True)), sub_trajectory[np.int(A.tolist().index(True)+1),0]),arrowprops = {'facecolor' : 'black'})
    
    plt.grid()
    plt.legend(fontsize=35)
    plt.tight_layout()
    plt.savefig(NOISE_TYPE+".jpeg", format='jpeg', dpi=300)
    plt.show()
