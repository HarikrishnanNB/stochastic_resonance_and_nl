
"""
Author: Harikrishnan NB
Email ID: harikrishnannb07@gmail.com
Dtd:  24- 02 -2021
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
import os
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.svm import LinearSVC

def skew_tent(x, b):
    """
    

    Parameters
    ----------
    x : TYPE : float
        DESCRIPTION-initial value of the skew tent map.
    b : TYPE : float
        DESCRIPTION-discrimination threshold of skew tent map.

    Returns
    -------
    TYPE : float
        DESCRIPTION-first iteration of the map.

    """
    if x < b:
        return x/b
    else:
        return (1-x)/(1-b)
    
    
def iterations(q, b, length):
    #The function return a time series and its index values 
    timeseries = (np.zeros((length, 1)))
    timeseries[0, 0] = q
    for inst in range(1, length):
        timeseries[inst, 0] = skew_tent(timeseries[inst-1, 0], b)
        
    return timeseries







def chaos_transform(X_train, timeseries, b, epsilon):
    
            
    M = X_train.shape[0]
    N = X_train.shape[1]

    firing_rate = np.zeros((M,N))
    firing_time = np.zeros((M,N))
    energy = np.zeros((M,N))
    entropy = np.zeros((M,N))
    
    
    for row in range(0,M):
        for col in range(0,N):
            
            A = (np.abs((X_train[row, col]) - timeseries[:,0]) < epsilon)
            # Firing Time
            firing_time[row, col] = A.tolist().index(True)
            freq = timeseries[0:A.tolist().index(True),0] - b < 0
            
            if len(freq) == 0:
                prob = 0
                firing_rate[row, col] = prob
                entropy[row, col] = 0
            else: 
                
                prob = freq.tolist().count(False)/len(freq)
                
                firing_rate[row, col] = prob
                if prob == 0 or prob == 1:
                    entropy[row, col] = 0
                else:
                    entropy[row, col] = -prob*np.log2(prob)-(1-prob)*np.log2(1-prob)
            
            x_chaotic = timeseries[0:A.tolist().index(True),0]
            # Energy of the chaotic trajectory
            
            energy[row, col] = np.sum(np.multiply(x_chaotic, x_chaotic))
            
    
    return np.concatenate((firing_rate, energy, firing_time, entropy),axis =1)
        

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



def k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, DISCRIMINATION_THRESHOLD, EPSILON, timeseries, DATA_NAME):
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
    ACCURACY = np.zeros((len(DISCRIMINATION_THRESHOLD), len(EPSILON)))
    FSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD),  len(EPSILON)))
    
    B = np.zeros((len(DISCRIMINATION_THRESHOLD),  len(EPSILON)))
    EPS = np.zeros((len(DISCRIMINATION_THRESHOLD),  len(EPSILON)))


    KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True) # Define the split - into 2 folds 
    KF.get_n_splits(traindata) # returns the number of splitting iterations in the cross-validator
    print(KF) 
    
    ROW = -1
    
   
    for DT in DISCRIMINATION_THRESHOLD:
        ROW = ROW+1
        COL = -1
       
        for EPSILON_1 in EPSILON:
             COL = COL+1
             ACC_TEMP =[]
             FSCORE_TEMP=[]
         
             for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                 
                 X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                 Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]
     
    
                 # Extract features X, q, b, epsilon, length
                 FEATURE_MATRIX_TRAIN = chaos_transform(X_TRAIN, timeseries, DT, EPSILON_1)
                 FEATURE_MATRIX_VAL = chaos_transform(X_VAL, timeseries, DT, EPSILON_1)            
             
                
                 mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN,Y_TRAIN, FEATURE_MATRIX_VAL)
                 
                 ACC = accuracy_score(Y_VAL, Y_PRED)*100
                 RECALL = recall_score(Y_VAL, Y_PRED , average="macro")
                 PRECISION = precision_score(Y_VAL, Y_PRED , average="macro")
                 F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
                              
                 
                 ACC_TEMP.append(ACC)
                 FSCORE_TEMP.append(F1SCORE)
            # Initial Neural Activity
             B[ROW, COL ] = DT # Discrimination Threshold
             EPS[ROW, COL ] = EPSILON_1 
             ACCURACY[ROW, COL ] = np.mean(ACC_TEMP)
             FSCORE[ROW, COL, ] = np.mean(FSCORE_TEMP)
             print("B = ", B[ROW, COL ],"EPSILON = ", EPS[ROW, COL]," is  = ",  np.mean(FSCORE_TEMP)  )
     
    print("Saving Hyperparameter Tuning Results")
    
       
    PATH = os.getcwd()
    RESULT_PATH = PATH + '/HR-PLOTS/'  + DATA_NAME + '/NEUROCHAOS-RESULTS/'
    
    
    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_PATH)
    
    np.save(RESULT_PATH+"/h_fscore.npy", FSCORE )    
    np.save(RESULT_PATH+"/h_accuracy.npy", ACCURACY )  
    np.save(RESULT_PATH+"/h_B.npy", B )
    np.save(RESULT_PATH+"/h_EPS.npy", EPS )               
    
    
    MAX_FSCORE = np.max(FSCORE)
    
    B_MAX = []
    EPSILON_MAX = []
    
    for ROW in range(0, len(DISCRIMINATION_THRESHOLD)):
        for COL in range(0, len(EPSILON)):
            
            if FSCORE[ROW, COL] == MAX_FSCORE:
                B_MAX.append(B[ROW, COL])
                EPSILON_MAX.append(EPS[ROW, COL])
    
    print("BEST F1SCORE", MAX_FSCORE)
  
    print("BEST DISCRIMINATION THRESHOLD = ", B_MAX)
    print("BEST EPSILON = ", EPSILON_MAX)
    return FSCORE, B, EPS, EPSILON
    
