# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:04:55 2021

@author: harik
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix as cm
# from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)


from chaos_codes import chaos_transform, chaosnet

from load_data import get_data

from hindmarsh_rose import hr_trajectory
# =============================================================================
# Parameters of Hindmarsh Rose Model in Chaotic Regime
# Refer the following paper for the parameters in chaotic regim.
# Holden, Arun V., and Yin-Shui Fan. "From simple to simple bursting oscillatory behaviour via chaos in the Rose-Hindmarsh model for neuronal activity." Chaos, Solitons & Fractals 2.3 (1992): 221-236.
# =============================================================================


X0 = [0, 0, 0] # Initial Conditions
current = 3.28

t = 2000
dt = 0.01
a = 1
b = 3
c = 1
d = 5
r = 0.0021
s = 4
x1 = -8/5
tvec = np.arange(0, t, dt)
x, y, z = hr_trajectory(X0, t, current, dt, r, x1, a, b, c, d, s)     
# =============================================================================
# Ploting the neuronal firing
# =============================================================================

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
f, ([ax1, ax2, ax3]) = plt.subplots(1, 3, figsize=(15, 3), tight_layout=True)

# Plot x(t)

ax1.plot(tvec, x, color='blue')
#ax1.set_title("x", fontsize=15)
ax1.set_xlabel("time (ms)", fontsize=20)
ax1.set_ylabel("x(t)", fontsize=20)
ax1.grid(alpha=0.3)

# Plot y(t)

ax2.plot(tvec, y, color='red')
#ax2.set_title("y", fontsize=15)
ax2.set_xlabel("time (ms)", fontsize=20)
ax2.set_ylabel("y(t)", fontsize=20)
ax2.grid(alpha=0.3)

# Plot z(t)

ax3.plot(tvec, z, color='green')
#ax3.set_title("z", fontsize=15)
ax3.set_xlabel("time (ms)", fontsize=20)
ax3.set_ylabel("z(t)", fontsize=20)
ax3.grid(alpha=0.3)
plt.show()
plt.savefig("chaos-chaos-transition.jpeg", format='jpeg', dpi=200)
plt.show()

## 3D plot
plt.figure(figsize=(15,10))
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
#mpl.rcParams['ztick.labelsize'] = label_size 
ax = plt.axes(projection='3d')
ax.set_xlabel("x(t)", fontsize=16)
ax.set_ylabel("y(t)", fontsize=16)
ax.set_zlabel("z(t)", fontsize=16)
# Data for a three-dimensional line

ax.plot3D(x, y, z, 'k')
plt.tight_layout()
#plt.savefig("3d_plot-chaos-chaos-transition.jpg", format='jpg', dpi=200)
