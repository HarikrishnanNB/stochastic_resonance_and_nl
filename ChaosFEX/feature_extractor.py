#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions for extracting features from a given 2D input
feature matrix by deriving estimates from paths taken by features along a
chaotic trajectory. Tuning parameters as well as hyperparameters are provided.

transform() is the main function that wraps around smaller modular functions
composed specifically for massive parallelization and performance optimizations
by Numba's JIT. The input 2D matrix with dimensions M x N expands to M x N*4.

Dependencies: numpy, numba

@author: Dr. Pranay S. Yadav
"""


# Import calls
import numpy as np
import numba as nb
import ChaosFEX.chaotic_sampler as cs
from ChaosFEX.input_validator import validate

# Pure python func with typing to check inequality for compiling as numpy ufunc
@nb.vectorize([nb.boolean(nb.float64, nb.float64, nb.float64)])
def _compare(value1, value2, value3):
    """
    This function calculates absolute distance (L1), checks whether it is
    less than epsilon and returns a corresponding boolean. It operates over
    scalar floats and is used by _compute_match_idx for speedy iteration.

    Parameters
    ----------
    value1 : scalar, float64
        A single value from the feature matrix.
    value2 : scalar, float64
        A single element from the trajectory array.
    value3 : scalar, float64
        The value epsilon.

    Returns
    -------
    bool
        True if the value (value1) from the feature matrix was within epsilon
        (value3) of the single element (value2) from trajectory array.

    """
    return abs(value1 - value2) < value3


# Check inequalities along a vector and terminate immediately upon match
@nb.njit
def _compute_match_idx(value, array, epsilon):
    """
    This function returns the index for which a given value comes within epsilon
    distance of any value in a given array, for the first time. Corresponds to
    a convergence to a neighborhood.

    Distance is evaluated by a dedicated function - _compare, that operates on
    scalars iteratively along the trajectory array.

    Parameters
    ----------
    value : scalar, float64
        A single value from the feature matrix.
    array : numpy array, 1D, float64
        Array containing values sampled from the trajectory of a chaotic map.
    epsilon : scalar, float64
        Distance for estimating approximation neighborhood while traversing
        along a chaotic trajectory.

    Returns
    -------
    int
        Index corresponding to the point along trajectory for which a value
        converges to within epsilon distance.

    """
    length = len(array)

    # Iterate over every element in the array
    for idx in range(length):

        # Check inequality
        if _compare(value, array[idx], epsilon):

            # Return index if match
            return idx

    # Exception: Failure of convergence
    # Return the length of the trajectory as we have traversed it fully
    return length


# Compute energy
@nb.njit
def _compute_energy(path):
    """
    This function computes the energy content of the path evaluated through a
    dot product with itself.

    Parameters
    ----------
    path : numpy array, 1D, float64
        DESCRIPTION.

    Returns
    -------
    scalar, float64
        Energy along the path traversed.

    """
    return path @ path


# Compute TTSS and entropy
@nb.njit
def _compute_ttss_entropy(path, threshold):
    """
    This function computes TTSS and Shannon Entropy based on the provided path.
    Threshold is used to bin the path into 2 values, from which probabilities
    are derived (TTSS). These are used to estimate entropy.

    Parameters
    ----------
    path : numpy array, 1D, float64
        DESCRIPTION.
    threshold : scalar, float64
        Threshold value of the skew-tent map.

    Returns
    -------
    2-element numpy array, 1D, float64
        1st element corresponds to TTSS
        2nd element corresponds to Shannon Entropy

    """
    prob = np.count_nonzero(path > threshold) / len(path)
    return np.array([prob, -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)])


@nb.njit(parallel=True)
def _compute_measures(feat_mat, trajectory, epsilon, threshold, meas_mat):
    """
    This functions iterates over elements in all rows and columns of the input
    feat_mat, computes 4 estimates and stores them in meas_mat along its 3rd
    dimension. Since meas_mat is initialized with 0s, any value not assigned
    is by default 0.

    Parameters
    ----------
    feat_mat : numpy array, 2D, float64
        Feature matrix of dimensions MxN, M are samples each with N features.
    trajectory : numpy array, 1D, float64
        Sampled trajectory along the skew-tent map.
    epsilon : scalar, float64
        Distance for estimating approximation neighborhood while traversing
        along a chaotic trajectory.
    threshold : scalar, float64
        Threshold value of the skew-tent map.
    meas_mat : numpy array, 3D, float64
        Zeros of shape MxNx4, 1st 2 dimensions correspond to those of
        feat_mat. The 3rd dimension has size 4, one for each feature estimated
        from the chaotic trajectory: TTSS, Energy, TT, & Entropy

    Returns
    -------
    meas_mat : numpy array, 3D, float64
        Contains computed estimates stored as follows:
            [i,j,0] : TTSS
            [i,j,1] : Energy
            [i,j,2] : TT/Steps/Index
            [i,j,3] : Entropy

    """
    # Iterate along rows
    for i in nb.prange(feat_mat.shape[0]):

        # Iterate along columns
        for j in nb.prange(feat_mat.shape[1]):

            # Compute index / TT corresponding to approximation / convergence
            idx = _compute_match_idx(feat_mat[i, j], trajectory, epsilon)
            meas_mat[i, j, 2] = idx

            # For non-zero index, compute the remaining measures
            if idx != 0:

                # Path traversed by value in element (i,j)
                path = trajectory[:idx]

                # Compute energy along path
                meas_mat[i, j, 1] = _compute_energy(path)

                # Compute TTSS and Entropy along path
                ttss_entropy = _compute_ttss_entropy(path, threshold)
                meas_mat[i, j, 0] = ttss_entropy[0]
                meas_mat[i, j, 3] = ttss_entropy[1]

    return meas_mat


def transform(feat_mat, initial_cond, trajectory_len, epsilon, threshold):
    """
    This function takes an input feature matrix with 4 tuning parameters
    for estimating features using a chaotic trajectory along the skew-tent map.
    Increases the feature space by 4-fold.

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
    out : numpy array, 2D, float64
        Contains computed estimates stored as follows:
            [i,[0,1]] : TTSS
            [i,[2,3]] : Energy
            [i,[4,5]] : TT/Steps/Index
            [i,[6,7]] : Entropy

    """
    # Stop if invalid inputs
    if not validate(feat_mat, initial_cond, trajectory_len, epsilon, threshold):
        return None

    # Initialize a 3D matrix of zeroes based on input dimensions
    dimx, dimy = feat_mat.shape
    meas_mat = np.zeros([dimx, dimy, 4])

    # Compute trajectory with specified parameters
    trajectory = cs.compute_trajectory(initial_cond, threshold, trajectory_len)

    # Estimate measures from the trajectory for each element in input matrix
    out = _compute_measures(feat_mat, trajectory, epsilon, threshold, meas_mat)

    # Convert nan's in entropy due to log(0) to 0s
    out[:, :, 3] = np.nan_to_num(out[:, :, 3])

    # Reshape 3D matrix to 2D with M x (N*4) dimensions and return
    out = out.transpose([0, 2, 1]).reshape([dimx, dimy * 4])
    return out


def warmup():
    """
    Warmup for initializing Numba's JIT compiler.
    Calls extract_feat with known and expected values.

    """
    # Initialize a feature_matrix
    feat_mat = np.array([[0.1, 0.2], [0.3, 0.4]])

    # Warmup the chaotic sampler
    cs.warmup()

    # Execute extract features
    out = transform(
        feat_mat, initial_cond=0.1, trajectory_len=100, epsilon=0.01, threshold=0.2
    )

    # Check if output matches expected value
    if out.shape == (2, 8) and out[0, 5] == 12:
        print("> Numba JIT warmup successful for transform ...")
    else:
        print("> Numba JIT warmup failed for transform ...")


# Execute warmup upon import
warmup()
