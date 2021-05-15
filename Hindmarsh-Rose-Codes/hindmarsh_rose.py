# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:23:09 2021

@author: Harikrishnan NB

Hindmarsh Rose model 

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



def hindmarsh_rose_model(X, t, current, r, x1, a, b, c, d, s):
    """
    

    Parameters
    ----------
    X : TYPE - 1D array
        DESCRIPTION: X[0], X[1], X[2] represents the initial condition of first, second and third variable of the Hindmarah Rose Model
    t : TYPE - 1D array
        DESCRIPTION: Time
    current : TYPE - scalar, float
        DESCRIPTION : External Input current
    r : TYPE - scalar, float
        DESCRIPTION : parameter of HR model
    x1 : TYPE - scalar, float
        DESCRIPTION : parameter of HR model
    a : TYPE - scalar, float
        DESCRIPTION : parameter of HR model
    b : TYPE - scalar, float
        DESCRIPTION : parameter of HR model
    c : TYPE - scalar, float
        DESCRIPTION : parameter of HR model
    d : TYPE - scalar, float
        DESCRIPTION : parameter of HR model
    s : TYPE - scalar, float
        DESCRIPTION : parameter of HR model

    Returns
    -------
    dxdt, dydt, dzdt
        DESCRIPTION.

    """
    return [X[1] - a * (X[0]**3) + b * (X[0]**2) - X[2] + current,
            c - d * (X[0]**2) - X[1],
                r * (s * (X[0] - x1) - X[2])]


def hr_trajectory(X0, t, current, dt, r, x1, a, b, c, d, s):
        """
    

    Parameters
    ----------
    X0 : TYPE - 1D array
        DESCRIPTION: initial condition
    t : TYPE - 1D array
        DESCRIPTION: time
    current : TYPE - scalar, float
        DESCRIPTION.
    dt : TYPE - scalar, float
        DESCRIPTION: timesteps

    Returns
    -------
    x : TYPE - 1D array
        DESCRIPTION : x(t) membrane potential (voltage)
    y : TYPE - 1D array
        DESCRIPTION:y(t) and z(t) are auxiliary variables describing, respectively, fast and slow transport processes
across the membrane.
    z : TYPE - 1D array
        DESCRIPTION: y(t) and z(t) are auxiliary variables describing, respectively, fast and slow transport processes
across the membrane.

    """
        
        tvec = np.arange(0, t, dt)
        X = odeint(hindmarsh_rose_model, X0, tvec, (current, r, x1, a, b, c, d, s))
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        return x, y, z


