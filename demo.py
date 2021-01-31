#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick script for demonstrating ChaosFEX functionality

@author: Dr. Pranay S. Yadav
"""

# Import calls
import numpy as np
import ChaosFEX.feature_extractor as CFX

# Initialize a 2D array
feat = np.random.random(size=(100,100))

# Extract features
CFX.transform(feat, 0.1, 1000, 0.1, 0.15)
