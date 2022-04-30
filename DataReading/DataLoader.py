import pandas as pd
import numpy as np
from numpy import double,loadtxt
from sklearn import preprocessing


def loadClasses(trainFile,testFile):
    trainStrY=loadStrLabel(trainFile)
    testStrY=loadStrLabel(testFile)
    uniqueStrY=np.array([e for e in np.unique(trainStrY) if e in np.unique(testStrY)])
    np.sort(uniqueStrY)
    classes=list()
    for i in range(len(uniqueStrY)):
        ascii=ord('A')+i
        if(ascii>90):
            ascii+=6
        classes.append(chr(ascii))
    return classes


def loadDataset(trainFile,testFile):
    trainX=loadScaledFeatures(trainFile)
    trainStrY=loadStrLabel(trainFile)
    testX=loadScaledFeatures(testFile)
    testStrY=loadStrLabel(testFile)
    instanceNum=len(trainX)
    trainX,trainStrY,testX,testStrY=labelChecker(trainX,trainStrY,testX,testStrY)
    trainY,testY=str2alpha(trainStrY,testStrY)
    return trainX,trainY,testX,testY,instanceNum


def loadStrLabel(filename):
    file_=open(filename,"r")
    line=file_.readline()
    # split
    line=line.strip()
    labels=line.split(',')
    file_.close()
    return labels


def loadScaledFeatures(filename):
    X=np.loadtxt(filename,skiprows=1,dtype=double,ndmin=2,delimiter=',')
    X=np.transpose(X)
    dataSet=preprocessing.scale(X)
    return dataSet

def str2alpha(trainStrY,testStrY):
    uniqueStrY=np.unique(trainStrY)
    np.sort(uniqueStrY)
    str2alpha_dict=dict()
    for i in range(len(uniqueStrY)):
        ascii=ord('A')+i
        if(ascii>90):
            ascii+=6
        str2alpha_dict[uniqueStrY[i]]=chr(ascii)
    
    trainY=list()
    testY=list()
    for i in range(len(trainStrY)):
        trainY.append(str2alpha_dict[trainStrY[i]])
    for j in range(len(testStrY)):
        testY.append(str2alpha_dict[testStrY[j]])
    
    trainY,testY=np.array(trainY),np.array(testY)
    return trainY,testY


def labelChecker(trainX,trainStrY,testX,testStrY):
    trainX=np.array(trainX)
    trainStrY=np.array(trainStrY)
    testX=np.array(np.array(testX))
    testStrY=np.array(testStrY)
    
    train_label_unique=np.unique(trainStrY)
    test_label_unique=np.unique(testStrY)
    useful_label_unique=np.array([e for e in train_label_unique if e in test_label_unique])
    #print(useful_label_unique)
    if len(train_label_unique)>len(useful_label_unique):
        useless_label=np.array([e for e in (set(train_label_unique)-set(useful_label_unique))])
        useless_index=np.array([i for i in range(len(trainStrY)) if trainStrY[i] in useless_label])
        
        # delete rows
        trainStrY=np.delete(trainStrY,useless_index,0)
        trainX=np.delete(trainX,useless_index,0)
        print("Useless labels in training samples:")
        print(useless_label)
        print("Data about labels above have been removed from training data.")
    
    if len(test_label_unique)>len(useful_label_unique):
        useless_label=np.array([e for e in (set(test_label_unique)-set(useful_label_unique))])
        useless_index=np.array([i for i in range(len(testStrY)) if testStrY[i] in useless_label])
        
        # delete rows
        testStrY=np.delete(testStrY,useless_index,0)
        testX=np.delete(testX,useless_index,0)
        print("Useless labels in testing samples:")
        print(useless_label)
        print("Data about labels above have been removed from testing data.")
    
    if len(train_label_unique)==len(useful_label_unique) and len(test_label_unique)==len(useful_label_unique):
        print("No error in training data and testing data.")
    return trainX,trainStrY,testX,testStrY

def loadData(trainFile, validateFile, testFile):
    train_X, train_y, validate_X, validate_y, instance_size = loadDataset(trainFile, validateFile)
    train_X, train_y, test_X, test_y, instance_size = loadDataset(trainFile, testFile)
    if np.unique(validate_y).shape[0] < np.unique(test_y).shape[0]:
        train_X, train_y, validate_X, validate_y, instance_size = loadDataset(trainFile, validateFile)
        target = np.unique(validate_y)
        test_X_y = pd.DataFrame(np.concatenate((test_X, np.array([test_y]).T), axis=1))
        result_X_y = test_X_y[test_X_y.iloc[:, -1].isin(target)]
        test_X = np.array(result_X_y.iloc[:, 0:-1])
        test_y = np.array(result_X_y.iloc[:, -1])
    elif np.unique(validate_y).shape[0] > np.unique(test_y).shape[0]:
        target = np.unique(test_y)
        validate_X_y = pd.DataFrame(np.concatenate((validate_X, np.array([validate_y]).T), axis=1))
        result_X_y = validate_X_y[validate_X_y.iloc[:, -1].isin(target)]
        validate_X = np.array(result_X_y.iloc[:, 0:-1])
        validate_y = np.array(result_X_y.iloc[:, -1])
    return train_X, train_y, validate_X, validate_y, test_X, test_y