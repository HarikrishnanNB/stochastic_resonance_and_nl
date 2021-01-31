# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:35:52 2021

@author: harik
"""

import os
import numpy as np
import pandas as pd
import scipy
from scipy.io import wavfile
from numpy.fft import fft
from sklearn.model_selection import train_test_split
import logging



def get_data(DATA_NAME):
    if DATA_NAME == "Jackson-speech":
        source = 'free-spoken-digit-dataset/free-spoken-digit-dataset-master/FSDD/'+DATA_NAME+'/'
        data_instances = len(os.listdir(source))
        
        
        labels = np.zeros((data_instances, 1), dtype='int')
        data_length = []
        
        for fileno, filename in enumerate(os.listdir(source)):
            
            sampling_frequency, data = wavfile.read(os.path.join(source,filename))
            data_length.append(len(data))
        
        input_features = np.min(data_length)
        
        fourier_data = np.zeros((data_instances, input_features))
        normal_data = np.zeros((data_instances, input_features))
        # Applying FFT 
        
        for fileno, filename in enumerate(os.listdir(source)):
            
            sampling_frequency, data = wavfile.read(os.path.join(source,filename))
            data_length.append(len(data))
            normal_data[fileno, :] = data[0:input_features]
            fourier_data[fileno, :] = np.abs(fft(data[0:input_features]))
            labels[fileno, 0] = filename[0]
        '''    
        if preprocessing == "fourier":
            DATA = fourier_data
        elif preprocessing == "no-preprocessing":
            DATA = normal_data
        '''
        DATA = fourier_data    
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(DATA, labels, test_size=0.2, random_state=21)
        X_TRAIN_NORM = ((X_TRAIN.T - np.min(X_TRAIN, axis = 1))/(np.max(X_TRAIN, axis= 1) - np.min(X_TRAIN, axis = 1))).T
        X_TEST_NORM = ((X_TEST.T - np.min(X_TEST, axis = 1))/(np.max(X_TEST, axis= 1) - np.min(X_TEST, axis = 1))).T
        print("Shape of Train data: ", X_TRAIN_NORM.shape)
        print("Shape of Test data: ", X_TEST_NORM.shape) 
        
        return X_TRAIN_NORM, Y_TRAIN, X_TEST_NORM, Y_TEST
    elif DATA_NAME == "concentric_circle":
        folder_path = "Data/" + DATA_NAME + "/" 
          
        # Load Train data
        X_train = np.array( pd.read_csv(folder_path+"X_train.csv", header = None) ) 
        # Load Train label
        trainlabel =  np.array( pd.read_csv(folder_path+"y_train.csv", header = None) )
        # Load Test data
        X_test = np.array( pd.read_csv(folder_path+"X_test.csv", header = None) )
        # Load Test label
        testlabel = np.array( pd.read_csv(folder_path+"y_test.csv", header = None) )
        
        ## Data_normalization - A Compulsory step
        # Normalization is done along each column
        
        X_train_norm = (X_train - np.min(X_train, 0))/(np.max(X_train, 0) - np.min(X_train, 0))
        X_test_norm = (X_test - np.min(X_test, 0))/(np.max(X_test, 0) - np.min(X_test, 0))

        try:
            assert np.min(X_train_norm) >= 0.0 and np.max(X_train_norm <= 1.0)
        except AssertionError:
            logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
 
        try:
            assert np.min(X_test_norm) >= 0.0 and np.max(X_test_norm <= 1.0)
        except AssertionError:
            logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
         
        return X_train_norm, trainlabel, X_test_norm, testlabel


    elif DATA_NAME == "concentric_circle_noise":
        folder_path = "Data/" + DATA_NAME + "/" 

        # Load Train data
        X_train = np.array( pd.read_csv(folder_path+"X_train.csv", header = None) ) 
        # Load Train label
        trainlabel =  np.array( pd.read_csv(folder_path+"y_train.csv", header = None) )
        # Load Test data
        X_test = np.array( pd.read_csv(folder_path+"X_test.csv", header = None) )
        # Load Test label
        testlabel = np.array( pd.read_csv(folder_path+"y_test.csv", header = None) )

        ## Data_normalization - A Compulsory step
        # Normalization is done along each column
        X_train_norm = (X_train - np.min(X_train, 0))/(np.max(X_train, 0) - np.min(X_train, 0))
        X_test_norm = (X_test - np.min(X_test, 0))/(np.max(X_test, 0) - np.min(X_test, 0))

        try:
            assert np.min(X_train_norm) >= 0.0 and np.max(X_train_norm <= 1.0)
        except AssertionError:
            logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
    
        try:
            assert np.min(X_test_norm) >= 0.0 and np.max(X_test_norm <= 1.0)
        except AssertionError:
            logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)

        return X_train_norm, trainlabel, X_test_norm, testlabel
    
    elif DATA_NAME == "single_variable_classification":
        t = 0*np.linspace(0,1,100)
        np.random.seed(42)
        class_0 = np.random.rand(100, 1) * 0.499
        np.random.seed(32)
        class_1 = np.random.rand(100, 1) * 0.499 + 0.5
        
        class_0_label = np.zeros((class_0.shape[0], 1))
        class_1_label = np.ones((class_1.shape[0], 1))
        
        traindata = np.concatenate((class_0, class_1))
        trainlabel = np.concatenate((class_0_label, class_1_label))
        
        np.random.seed(64)
        class_0_test = np.random.rand(100, 1) * 0.499
        np.random.seed(68)
        class_1_test = np.random.rand(100, 1) * 0.499 + 0.5
        
        class_0_testlabel = np.zeros((class_0_test.shape[0], 1))
        class_1_testlabel = np.ones((class_1_test.shape[0], 1))
        
        testdata = np.concatenate((class_0_test, class_1_test))
        testlabel = np.concatenate((class_0_testlabel, class_1_testlabel))
        
        PATH = os.getcwd()
        RESULT_PATH = PATH + '/Data/single_variable_classification/'
        
        
        try:
            os.makedirs(RESULT_PATH)
        except OSError:
            print ("Creation of the result directory %s failed" % RESULT_PATH)
        else:
            print ("Successfully created the result directory %s" % RESULT_PATH)
        
        np.save(RESULT_PATH+"/testdata.npy", testdata)    
        np.save(RESULT_PATH+"/testlabel.npy", testlabel) 
        np.save(RESULT_PATH+"/traindata.npy", traindata) 
        np.save(RESULT_PATH+"/trainlabel.npy", trainlabel)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,10))
        plt.plot(class_0_test,t,'*k', markersize = 10, label='Class-0')
        plt.plot(class_1_test,t,'or', markersize = 10, label='Class-1')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.xlabel('$f_1$', fontsize=20)
        plt.ylabel('$f_2$', fontsize=20)
        plt.ylim(-0.5, 0.5)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(RESULT_PATH+DATA_NAME+"-data.jpg", format='jpg', dpi=200)
        plt.show()

        
        return traindata, trainlabel, testdata, testlabel