#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions for validation of various input arguments to functions in chaotic_sampler.py & feature_extractor.py

validate() is the main function that wraps around smaller modular functions.

Dependencies: numpy

@author: Dr. Pranay S. Yadav
"""


# Import calls
from numpy import ndarray, float64

# Function definitions
def _check_trajectory_inputs(init_cond, threshold, trajectory_len):
    """
    This function checks for the type and range of the 3 hyperparameters for
    the skew-tent map. These are the input to the function compute_trajectory
    from the module chaotic_sampler.py

    Parameters
    ----------
    init_cond : scalar, float64
        Initial value for iterating through the skew-tent map.
            range: 0 < init_cond < 1
    threshold : scalar, float64
        Threshold value of the skew-tent map.
            range: 0 < threshold < 1
    trajectory_len : scalar, integer
        Size of the trajectory to compute through iteration.
            range: 10^2 < length < 10^7

    Returns
    -------
    bool
        DESCRIPTION.

    """
    # Check types of init_cond and threshold
    if not (isinstance(init_cond, float) and isinstance(threshold, float)):
        print("> ERROR: init_cond & threshold should be of type float ...")
        return False

    # Check ranges of init_cond and threshold
    if not (0 <= init_cond <= 1 and 0 <= threshold <= 1):
        print("> ERROR: init_condition & threshold cannot be <=0 or >=1 ...")
        return False

    # Check type & range of length
    if not (100 <= trajectory_len <= int(1e7) and isinstance(trajectory_len, int)):
        print("> ERROR: length should be an integer between 10^2 & 10^7 ...")
        return False

    return True


def _check_features(feat_mat):
    """
    This function checks for the type, dimensions and scaling of the input.
    Expected input is the feature matrix with dimensions MxN, where M is the
    number of samples and N is the number features per sample.

    This matrix (2D array) is the primary data input to extract_feat in module
    feature_extractor.py

    Parameters
    ----------
    feat_mat : numpy array, 2D, float64
        Feature matrix of dimensions MxN, M are samples each with N features.

    Returns
    -------
    bool
        Validity of input.

    """
    # Check type and shape of input feature matrix
    if not (
        isinstance(feat_mat, ndarray)
        and feat_mat.dtype == float64
        and feat_mat.ndim == 2
    ):
        print("> ERROR: feat_mat should be 2D array of dtype float64 ...")
        return False

    # Check ranges of values in input feature matrix
    if feat_mat.min() < 0 or feat_mat.max() > 1:
        print("> ERROR: feat_mat should be scaled between 0 & 1 ...")
        return False

    return True


def _check_epsilon(epsilon):
    """
    This function checks for the type and bounds of the convergence parameter
    epsilon for determining neighborhood approximation.

    The parameter epsilon is a tuning parameter for convergence of the function
    extract_feat in module feature_extractor.py

    Parameters
    ----------
    epsilon : scalar, float
        Distance for estimating approximation neighborhood while traversing
        along a chaotic trajectory. Value should lie between suggested
        heuristic bounds of 0.3 and 10^-5.

    Returns
    -------
    bool
        Validity of input.

    """
    if not (isinstance(epsilon, float) and 1e-5 <= epsilon <= 1.0):
        print("> ERROR: epsilon must be a float between 0.5 and 10^-5")
        return False

    return True


def validate(feat_mat, initial_cond, trajectory_len, epsilon, threshold):
    """
    This function is a wrapper around _check_trajectory_inputs, _check_features,
    and _check_epsilon. It checks for all the inputs passed to the function
    extract_feat in module feature_extractor.py

    Parameters
    ----------
    feat_mat : numpy array, 2D, float64
        Feature matrix of dimensions MxN, M are samples each with N features.
    initial_cond : scalar, float64
        Initial value for iterating through the skew-tent map.
            range: 0 < init_cond < 1
    trajectory_len : scalar, integer
        Size of the trajectory to compute through iteration.
            range: 10^2 < length < 10^7
    epsilon : scalar, float
        Distance for estimating approximation neighborhood while traversing
        along a chaotic trajectory. Value should lie between suggested
        heuristic bounds of 0.3 and 10^-5.
    threshold : scalar, float64
        Threshold value of the skew-tent map.
            range: 0 < threshold < 1

    Returns
    -------
    bool
        DESCRIPTION.

    """
    if (
        _check_epsilon(epsilon)
        and _check_features(feat_mat)
        and _check_trajectory_inputs(initial_cond, threshold, trajectory_len)
    ):
        return True
    else:
        return False
