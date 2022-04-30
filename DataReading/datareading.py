import numpy as np
import pandas as pd


def get_data(file_path):
    """to get the data which is divided into train,test,validation
    the first row is label
    and then every column is a sample
    :returns
    feature:every row is a sample
    label:label for each sample
    """
    file = open(file_path, "r")
    line = file.readline().strip('\n')
    label = line.split(',')
    label = np.array(label)
    feature = np.empty(shape=[0, label.shape[0]])
    while line != "":
        line = file.readline().strip('\n')
        if line != "":
            sub_feature = line.split(',')
            sub_feature = list(map(lambda x: float(x), sub_feature))
            feature = np.row_stack((feature, np.array([sub_feature])))
    feature = np.transpose(feature)
    return feature, label


def get_csv(file_path):
    """
    description:
        to get the data which is a big csv,include train,test,validation
        divide the data to train,test,validation which are the same distribution
    return:
        train_X,validate_X,test_X: a row is a sample
        train_y,validate_y,test_y: label for each sample
    """
    data = pd.read_csv(file_path, header=None)
    group = data.groupby(data[data.columns[-1]])
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    validate_df = pd.DataFrame()
    for g in group:
        count_sample = g[1].shape[0]
        if count_sample <= 2:
            continue
        else:
            train_df = pd.concat([train_df, g[1][:int(count_sample/2)]])
            validate_df = pd.concat([validate_df, g[1][int(count_sample/2):int(count_sample*3/4)]])
            test_df = pd.concat([test_df, g[1][int(count_sample*3/4):count_sample+1]])
    train_y = np.array(train_df.iloc[:, -1])
    train_X = np.array(train_df.iloc[:, 0:-1])
    validate_y = np.array(validate_df.iloc[:, -1])
    validate_X = np.array(validate_df.iloc[:, 0:-1])
    test_y = np.array(test_df.iloc[:, -1])
    test_X = np.array(test_df.iloc[:, 0:-1])
    return train_X, train_y, validate_X, validate_y, test_X, test_y


'''filepath = "D:\ECOC\data_uci\zoo_train.data"
X, y = get_data(filepath)
print(X, y)

filepath = "D:\ECOC\\test_set\\zoo.csv"
get_csv(filepath)
'''