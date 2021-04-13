# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Dtd: 22 Dec. 2020
ChaosNet decision function
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
import os
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.svm import LinearSVC

import ChaosFEX.feature_extractor as CFX


def chaosnet(traindata, trainlabel, testdata):
    '''
    

    Parameters
    ----------
    traindata : TYPE - Numpy 2D array
        DESCRIPTION - traindata
    trainlabel : TYPE - Numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE - Numpy 2D array
        DESCRIPTION - testdata

    Returns
    -------
    mean_each_class : Numpy 2D array
        DESCRIPTION - mean representation vector of each class
    predicted_label : TYPE - numpy 1D array
        DESCRIPTION - predicted label

    '''
    from sklearn.metrics.pairwise import cosine_similarity
    NUM_FEATURES = traindata.shape[1]
    NUM_CLASSES = len(np.unique(trainlabel))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        
        mean_each_class[label, :] = np.mean(traindata[(trainlabel == label)[:,0], :], axis=0)
        
    predicted_label = np.argmax(cosine_similarity(testdata, mean_each_class), axis = 1)

    return mean_each_class, predicted_label




def k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON, DATA_NAME):
    """

    Parameters
    ----------
    FOLD_NO : TYPE-Integer
        DESCRIPTION-K fold classification.
    traindata : TYPE-numpy 2D array
        DESCRIPTION - Traindata
    trainlabel : TYPE-numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE-numpy 2D array
        DESCRIPTION - Testdata
    testlabel : TYPE - numpy 2D array
        DESCRIPTION - Testlabel
    INITIAL_NEURAL_ACTIVITY : TYPE - numpy 1D array
        DESCRIPTION - initial value of the chaotic skew tent map.
    DISCRIMINATION_THRESHOLD : numpy 1D array
        DESCRIPTION - thresholds of the chaotic map
    EPSILON : TYPE numpy 1D array
        DESCRIPTION - noise intenity for NL to work (low value of epsilon implies low noise )
    DATA_NAME : TYPE - string
        DESCRIPTION.

    Returns
    -------
    FSCORE, Q, B, EPS, EPSILON

    """
    ACCURACY = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    FSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    Q = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    B = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    EPS = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))


    KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True) # Define the split - into 2 folds 
    KF.get_n_splits(traindata) # returns the number of splitting iterations in the cross-validator
    print(KF) 
    
    ROW = -1
    COL = -1
    WIDTH = -1
    for DT in DISCRIMINATION_THRESHOLD:
        ROW = ROW+1
        COL = -1
        WIDTH = -1
        for INA in INITIAL_NEURAL_ACTIVITY:
            COL =COL+1
            WIDTH = -1
            for EPSILON_1 in EPSILON:
                WIDTH = WIDTH + 1
                
                ACC_TEMP =[]
                FSCORE_TEMP=[]
            
                for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                    
                    X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                    Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]
        

                    # Extract features
                    FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                    FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)            
                
                   
                    mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN,Y_TRAIN, FEATURE_MATRIX_VAL)
                    
                    ACC = accuracy_score(Y_VAL, Y_PRED)*100
                    RECALL = recall_score(Y_VAL, Y_PRED , average="macro")
                    PRECISION = precision_score(Y_VAL, Y_PRED , average="macro")
                    F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
                                 
                    
                    ACC_TEMP.append(ACC)
                    FSCORE_TEMP.append(F1SCORE)
                Q[ROW, COL, WIDTH ] = INA # Initial Neural Activity
                B[ROW, COL, WIDTH ] = DT # Discrimination Threshold
                EPS[ROW, COL, WIDTH ] = EPSILON_1 
                ACCURACY[ROW, COL, WIDTH ] = np.mean(ACC_TEMP)
                FSCORE[ROW, COL, WIDTH ] = np.mean(FSCORE_TEMP)
                print("Mean F1-Score for Q = ", Q[ROW, COL, WIDTH ],"B = ", B[ROW, COL, WIDTH ],"EPSILON = ", EPS[ROW, COL, WIDTH ]," is  = ",  np.mean(FSCORE_TEMP)  )
    
    print("Saving Hyperparameter Tuning Results")
    
    
    PATH = os.getcwd()
    RESULT_PATH = PATH + '/SR-PLOTS/'  + DATA_NAME + '/NEUROCHAOS-RESULTS/'
    
    
    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_PATH)
    
    np.save(RESULT_PATH+"/h_fscore.npy", FSCORE )    
    np.save(RESULT_PATH+"/h_accuracy.npy", ACCURACY ) 
    np.save(RESULT_PATH+"/h_Q.npy", Q ) 
    np.save(RESULT_PATH+"/h_B.npy", B )
    np.save(RESULT_PATH+"/h_EPS.npy", EPS )               
    
    
    MAX_FSCORE = np.max(FSCORE)
    Q_MAX = []
    B_MAX = []
    EPSILON_MAX = []
    
    for ROW in range(0, len(DISCRIMINATION_THRESHOLD)):
        for COL in range(0, len(INITIAL_NEURAL_ACTIVITY)):
            for WID in range(0, len(EPSILON)):
                if FSCORE[ROW, COL, WID] == MAX_FSCORE:
                    Q_MAX.append(Q[ROW, COL, WID])
                    B_MAX.append(B[ROW, COL, WID])
                    EPSILON_MAX.append(EPS[ROW, COL, WID])
    
    print("BEST F1SCORE", MAX_FSCORE)
    print("BEST INITIAL NEURAL ACTIVITY = ", Q_MAX)
    print("BEST DISCRIMINATION THRESHOLD = ", B_MAX)
    print("BEST EPSILON = ", EPSILON_MAX)
    return FSCORE, Q, B, EPS, EPSILON
    

