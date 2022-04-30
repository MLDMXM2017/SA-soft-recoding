import copy
from CodeMatrix import CodeMatrix
from DataReading.DataLoader import *
from Classifiers.ECOCClassifier import SimpleECOCClassifier
from Classifiers.BaseClassifier import get_base_clf
from CorrectRate.correctrate import Indicator
from sklearn.impute import SimpleImputer
from Loss.Loss import Loss

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from multiprocessing import Process, Pool
import pandas as pd

import platform
from ECOC_kit import Classifier
import time
import warnings
warnings.filterwarnings("ignore")


def get_codematrix(train_X, train_y, matrix_type="dense"):
    if matrix_type == "dense":
        return CodeMatrix.dense_rand(train_X, train_y)
    elif  matrix_type == "sparse":
        return CodeMatrix.sparse_rand(train_X, train_y)
    elif  matrix_type == "ova":
        return CodeMatrix.ova(train_X, train_y)
    elif matrix_type == "decoc":
        return Classifier.D_ECOC().create_matrix(train_X, train_y)
    else:
        return CodeMatrix.ovo(train_X, train_y)


def norm_data(data, y):
    data = SimpleImputer().fit_transform(data)
    data = preprocessing.StandardScaler().fit_transform(data, y)
    # data = preprocessing.scale(data)
    return data


def test(dataset, fold, ecoc_type, alpha):
    file_path = base_folder + dataset + '.csv'
    df = pd.read_csv(file_path, header=None, index_col=None)
    X = df.values[:, :-1]
    y = df.values[:, -1]
    X = norm_data(X, y)
    train_random_gradient_indicators = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1-score'])
    classes = np.unique(y)
    for t in range(repeat):
        kf = KFold(n_splits=fold)
        accs = []
        fscores = []
        for train_idx, test_idx in kf.split(X, y):
            full_train_X, test_X = X[train_idx, :], X[test_idx, :]
            full_train_y, test_y = y[train_idx], y[test_idx]
            estimator = get_base_clf(base_learner)
            random_code_matrix, index = get_codematrix(X, y, ecoc_type)
            sec = SimpleECOCClassifier(y, estimator, random_code_matrix, Decoder)
            sec.fit(full_train_X, full_train_y)
            preditct_Y = sec.predict_y(full_train_X)
            loss = Loss(classes, copy.deepcopy(random_code_matrix))
            code_matrix = loss.update_code_matrix_by_row(preditct_Y, full_train_y, sec, alpha, T, ε)
            sec.set_code_matrix(code_matrix)
            train_pred, train_gr_distances, Y = sec.predict(test_X)
            train_random_gradient_indicators.loc[t] = Indicator.get_indicators(train_pred, test_y)
            accs.append(train_random_gradient_indicators.loc[t][0])
            fscores.append(train_random_gradient_indicators.loc[t][3])
            print("tr_soft_gr = {}".format(np.around(code_matrix, decimals=3)))
            print("tr_soft_gr = {}".format(train_random_gradient_indicators.loc[t]))


datasets = ['Breast']
repeat = 10
fold = 5
alpha_list = [0.01]
Decoder = 'ED'
base_learner = 'SVM'
T = 500
ε = 0.00001
ecoc_types = ['dense']

base_folder = './data/'
if __name__ == '__main__':
    is_windows = platform.system() == "Windows"
    if not is_windows:
        pool = Pool()
    for ecoc_type in ecoc_types:
        for real_alpha in alpha_list:
            for dataset in datasets:
                # dataset = dataset + ".csv"
                if is_windows:
                    test(dataset, fold, ecoc_type, real_alpha)
                else:
                    future = pool.apply_async(test, args=(dataset, fold, ecoc_type, real_alpha))
    if not is_windows:
        pool.close()
        pool.join()


