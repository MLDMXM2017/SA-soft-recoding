import pandas as pd
import numpy as np
import warnings
from Decoding.Decoder import get_decoder
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import copy
import math


def _check_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if (not hasattr(estimator, "decision_function") and
            not hasattr(estimator, "predict_proba")):
        raise ValueError("The base estimator should implement "
                         "decision_function or predict_proba!")


def check_is_fitted(estimator, attributes):
    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]


def _fit_ternary(estimator, X, y):
    """Fit a single ternary estimator. not offical editing.
        delete item from X and y when y = 0
        edit by elfen.
    """
    t, l = X[y != 0], y[y != 0]
    unique_y = np.unique(l)
    if len(unique_y) == 1:
        warnings.warn('only one class')
    
    estimator = copy.deepcopy(estimator)
    estimator.fit(t, l)
    return estimator


def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor."""
    return getattr(estimator, "_estimator_type", None) == "regressor"


def _predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    if is_regressor(estimator):
        return estimator.predict(X)
    try:
        temp = estimator.predict_proba(X)
        k = np.argmax(temp, axis=1)
        score = estimator.classes_[k] * (np.choose(k, temp.T) - 0.5) * 2
        # proba = estimator.predict_proba(X)
        # mask = 2 * np.argmax(proba, axis=1) - 1
        # score = np.max(proba, axis=1) * mask
        # score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        # probabilities of the positive class
        proba = estimator.predict_proba(X)
        mask = 2 * np.argmax(proba, axis=1) - 1
        score = np.max(proba, axis=1) * mask
    return score

def _predict_binary2(estimator, X):
    """Make predictions using a single binary estimator."""
    return estimator.predict(X)



def _sigmoid_normalize(X):
    return 1 / (1 + np.exp(-X))


def _min_max_normalize(X):
    """Min max normalization
    warning: 0 value turns not 0 in most cases.
    """
    res = []
    for x in X.T:
        x_min, x_max = min(x), max(x)
        x_range = x_max - x_min
        if x_range == 0:
            res.append(x)
        else:
            res.append([float(i-x_min)/x_range for i in x])
    return np.array(res).T


class SimpleECOCClassifier:
    """ A simple ECOC classifier
    Parameters:
        estimator: object
            unfitted base classifier object.
        code_matrix: 2-d array
            code matrix (Classes×Dichotomies).
        decoder: str
            indicates the type of decoder, get a decoder object immediately when initialization.
            For more details, check Decoding.Decoder.get_decoder.
        soft: bool, default True.
            Whether to use soft distance to decode.

    Attributes:
        estimator_type: str, {'decision_function','predict_proba'}
            which type the estimator belongs to.
            'decision_function' - predict value range (-∞,+∞)
            'predict_proba' - predict value range [0,1]
        classes_: set
            the set of labels.
        estimators_: 1-d array
            trained classifers.

    Methods:
        fit(X, y): Fit the model according to the given training data.
        predict(X): Predict class labels for samples in X.
        fit_predict(X, y, test_X): fit(X, y) then predict(X_test).

    Descriptions:
        fit(X, y): Fit the model according to the given training data.
            Parameters:
                X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Training vector, where n_samples in the number of samples and n_features is the number of features.
                y: array-like, shape = [n_samples]
                    Target vector relative to X
            Returns:
                self: object
                    Returns self.

        predict(X): Predict class labels for samples in X.
            Parameters:
                X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Samples.
            Returns:
                C: array, shape = [n_samples]
                    Predicted class label per sample.

        fit_predict(X, y, test_X): fit(X, y) then predict(X_test).
            Parameters:
                X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Training vector, where n_samples in the number of samples and n_features is the number of features.
                y: array-like, shape = [n_samples]
                    Target vector relative to
                X_test: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Samples.
            Returns:
                C: array, shape = [n_samples]
                    Predicted class label per sample.
            Notes: This is a combination of two methods fit & predict, with X, y for fit and X_test for predict.
                Run fit first and then run predict
    """
    def __init__(self, y, estimator, code_matrix, decoder='AED', soft=True):
        self.estimator = estimator  # classifier
        self.code_matrix = code_matrix  # code matrix
        self.decoder = get_decoder(decoder)  # decoder
        self.soft = soft  # if using soft distance.
        self.classes_ = np.unique(y)
        # self.p = np.ones((self.code_matrix.shape[0], self.code_matrix.shape[1]))
        self.us = None
        self.ts = None
        self.W = np.ones((self.code_matrix.shape[0], self.code_matrix.shape[1]))
        self.flag = True
    
    def update_performance_matrix(self, predict_result, labels):
        labels = np.array(labels)
        # distances = self.decoder.decode(predict_result, self.code_matrix, self.us, self.ts)
        self.us = np.ones((self.code_matrix.shape[0], self.code_matrix.shape[1]))
        self.ts = np.ones((self.code_matrix.shape[0], self.code_matrix.shape[1]))
        for i in range(self.code_matrix.shape[0]):
            ith_result = predict_result[labels == i, :] 
            us = np.mean(ith_result, axis=0)
            ts = np.std(ith_result, axis=0)
            self.us[i, :] = us
            self.ts[i, :] = ts
        self.flag = False

    def set_new_decode(self, decoder):
        self.decoder = get_decoder(decoder)

    def fit(self, X, y, estimators=None):
        #_check_estimator(self.estimator)
        if hasattr(self.estimator, "decision_function"):
            self.estimator_type = 'decision_function'
        elif hasattr(self.estimator, "predict_proba"):
            self.estimator_type = 'predict_proba'

        
        classes_index = dict((c, i) for i, c in enumerate(self.classes_))
        Y = np.array([self.code_matrix[classes_index[y[i]]] for i in range(X.shape[0])])
        
        if estimators:
            self.estimators_ = estimators
        else:
            self.estimators_ = [_fit_ternary(self.estimator, X, Y[:, i]) for i in
                                range(Y.shape[1])]
        # pred_X = np.array([_predict_binary(self.estimators_[i], X) for i in range(len(self.estimators_))]).T
        # self.scaler_ = [MinMaxScaler().fit(pred_X[:, i]) for i in range(len(self.estimators_))]
        return self

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        Y = np.array([_predict_binary(self.estimators_[i], X) for i in range(len(self.estimators_))]).T
        
            
        
        # Y = np.array([self.scaler_[i].transform(Y[:, i]) for i in range(len(self.scaler_))])
        # Y_min, Y_max = Y.min(), Y.max()
        # print('%s: (%f , %f)' % (self.estimator_type, Y_min, Y_max))

        #if self.estimator_type == 'decision_function':
        # Y = _min_max_normalize(Y)  # Use a normalization because scale of Y is [-1,1]

        # Y = Y * 2 - 1  # mapping scale [0, +1] to [-1, +1]

        # Y = np.tanh(Y)
        if self.flag:
            distances = self.decoder.decode(Y, self.code_matrix, self.us, self.ts)
       
            pred = distances.argmin(axis=1)
        else:
            pred = [self._vector_score_interval(y) for y in Y]
            distances = None
        
        return self.classes_[pred], distances, Y
    
       

    def fit_predict(self, X, y, test_X):
        self.fit(X, y)
        return self.predict(test_X)

    """Edit by ZengNan"""
    def set_code_matrix(self, code_matrix):
        """"modify self.code_matrix"""
        self.code_matrix = code_matrix

    def predict_y(self, X):
        """get the vector the classifiers predict
        :returns
        the vectors"""
        check_is_fitted(self, 'estimators_')
        Y = np.array([_predict_binary(self.estimators_[i], X) for i in range(len(self.estimators_))]).T
        # Y = np.array([self.scaler_[i].transform(Y[:, i]) for i in range(len(self.scaler_))])
        # Y = 
        # Y = _min_max_normalize(Y)  # Use a normalization because scale of Y is [-1,1]
        # Y = np.tanh(Y)
        # Y = Y * 2 - 1  # mapping scale [0, +1] to [-1, +1]
        return Y
    
    def _vector_score_interval(self, vector):
        """
        calculate the score of the output vector to each class (using soft value interval method)
        
        Parameters:
            vector: output vector of the base leaners
            interval_matrix: the soft interval matrix
            weight_vector: the weight vector of each base leaners
            soft_matrix: the soft (mean) values matrix
        Returns: 
            classifier result, the class index which has the highest score
        """
        index = None
        scores = [0] * self.us.shape[0]
        # distances = []
        for i in range(self.us.shape[0]):
            for j in range(len(self.us[i])):
                if vector[j] >= (self.us[i,j] - self.ts[i,j]) and vector[j] <= (self.us[i,j] + self.ts[i,j]):
                    scores[i] += 1 * math.exp(abs(vector[j])-1)
                else:
                    distance = abs(vector[j] - self.code_matrix[i][j])
                    # distances.append(distance)
                    try:
                        scores[i] += math.exp(-distance) * math.exp(abs(vector[j])-1)
                    except OverflowError:
                        scores[i] = 0
            index = np.argmax(scores)
        return index


    def soft_predict(self, X, soft_code_matrix):
        """use soft code matrix to predict the label
        :return labels"""
        Y = self.predict_y(X)
        distances = self.decoder.decode(Y, soft_code_matrix, self.us, self.ts)
        pred = distances.argmin(axis=1)
        #print(pred)
        #print(self.classes_)
        #print(self.code_matrix.shape)
        return self.classes_[pred], distances, Y

    def adaboost_fit(self, X, y, all = True):
        """"over sampling
        if all, over sampling all samples
        else, over sampling wrong samples"""
        _check_estimator(self.estimator)
        if hasattr(self.estimator, "decision_function"):
            self.estimator_type = 'decision_function'
        else:
            self.estimator_type = 'predict_proba'

        self.classes_ = np.unique(y)
        classes_index = dict((c, i) for i, c in enumerate(self.classes_))
        Y = np.array([self.code_matrix[classes_index[y[i]]] for i in range(X.shape[0])], dtype=np.int)
        if all:
            self.adaboost_all_fit_emstimator(X, Y, y, classes_index)
        else:
            self.adaboost_wrong_fit_emstimator(X, Y, y, classes_index)
        return self

    def fit_expand(self, X, y):
        """test the fit"""
        _check_estimator(self.estimator)
        if hasattr(self.estimator, "decision_function"):
            self.estimator_type = 'decision_function'
        else:
            self.estimator_type = 'predict_proba'

        self.classes_ = np.unique(y)
        classes_index = dict((c, i) for i, c in enumerate(self.classes_))
        Y = np.array([self.code_matrix[classes_index[y[i]]] for i in range(X.shape[0])], dtype=np.int)
        predict_code_matrix = Y
        self.estimators_ = []
        for m in range(Y.shape[1]):
            estimator = _fit_ternary(self.estimator, X, Y[:, m])
            self.estimators_.append(estimator)
            for n in range(len(self.estimators_)):
                predict_code_matrix[:, n] = _predict_binary(self.estimators_[n], X)
            pre_y = self.decoder.decode(predict_code_matrix, self.code_matrix).argmin(axis=1)
            group_y = pd.DataFrame(y).groupby(y)
            start = 0
            accuracy = []
            for i in group_y:
                end = start + i[1].shape[0]
                count = 0
                for j in range(start, end):
                    if classes_index[i[1].iloc[j - start].values[0]] == pre_y[i[1].iloc[j - start].name]:
                        count += 1
                accuracy.append(count / (end - start))
                start = end
            #print(accuracy)
        return self

    def adaboost_all_fit_emstimator(self, X, Y, y, classes_index):
        """over sampling all samples"""
        predict_code_matrix = Y
        self.estimators_ = []
        for m in range(Y.shape[1]):
            estimator = _fit_ternary(self.estimator, X, Y[:, m])
            self.estimators_.append(estimator)
            for n in range(len(self.estimators_)):
                predict_code_matrix[:, n] = _predict_binary(self.estimators_[n], X)
            pre_y = self.decoder.decode(predict_code_matrix, self.code_matrix).argmin(axis=1)
            group_y = pd.DataFrame(y).groupby(y)
            start = 0
            accuracy = []
            for i in group_y:
                end = start + i[1].shape[0]
                count = 0
                for j in range(start, end):
                    if classes_index[i[1].iloc[j - start].values[0]] == pre_y[i[1].iloc[j-start].name]:
                        count += 1
                accuracy.append(count/(end-start))
                start = end
            #print(accuracy)
            minaccuracy = min(accuracy)
            index_of_target_class = accuracy.index(minaccuracy)
            target_class = self.classes_[index_of_target_class]
            #print(target_class)
            if minaccuracy == 1.0:
                continue
            else:
                X_y = np.concatenate((X, np.array([y]).T), axis=1)
                Increase_X = X_y[X_y[:, -1] == target_class, 0:-1]
                X = np.concatenate((X, Increase_X), axis=0)
                y = np.concatenate((y, X_y[X_y[:, -1] == target_class, -1]), axis=0)
                Increase_Y = np.tile(self.code_matrix[index_of_target_class], (Increase_X.shape[0], 1))
                Y = np.concatenate((Y, Increase_Y), axis=0)
                predict_code_matrix = Y
                for n in range(len(self.estimators_)):
                    predict_code_matrix[:, n] = _predict_binary(self.estimators_[n], X)

    def adaboost_wrong_fit_emstimator(self, X, Y, y, classes_index):
        """over sampling wrong samples"""
        predict_code_matrix = Y
        self.estimators_ = []
        for m in range(Y.shape[1]):
            estimator = _fit_ternary(self.estimator, X, Y[:, m])
            self.estimators_.append(estimator)
            for n in range(len(self.estimators_)):
                predict_code_matrix[:, n] = _predict_binary(self.estimators_[n], X)
            pre_y = self.decoder.decode(predict_code_matrix, self.code_matrix).argmin(axis=1)
            group_y = pd.DataFrame(y).groupby(y)
            start = 0
            accuracy = []
            Increase_X = np.empty(shape=[0, X.shape[1]])
            Increase_y = np.array([],dtype=object)
            for i in group_y:
                end = start + i[1].shape[0]
                count = 0
                for j in range(start, end):
                    if classes_index[i[1].iloc[j - start].values[0]] == pre_y[i[1].iloc[j-start].name]:
                        count += 1
                    else:
                        Increase_X = np.concatenate((Increase_X,
                                                     np.array([X[i[1].iloc[j-start].name]])))
                        Increase_y = np.hstack((Increase_y, i[1].iloc[j - start].values[0]))
                accuracy.append(count/(end-start))
                start = end
            #print(accuracy)
            minaccuracy = min(accuracy)
            index_of_target_class = accuracy.index(minaccuracy)
            target_class = self.classes_[index_of_target_class]
            #print(target_class)
            if minaccuracy == 1.0:
                continue
            else:
                Increase_X_y = np.concatenate((Increase_X, np.array([Increase_y]).T), axis=1)
                X = np.concatenate((X, Increase_X_y[Increase_X_y[:, -1] == target_class, 0:-1]), axis=0)
                y = np.concatenate((y, Increase_X_y[Increase_X_y[:, -1] == target_class, -1]), axis=0)
                Increase_Y = np.tile(self.code_matrix[index_of_target_class],
                                     (Increase_X_y[Increase_X_y[:, -1] == target_class, 0:-1].shape[0], 1))
                Y = np.concatenate((Y, Increase_Y), axis=0)
                predict_code_matrix = Y
                for n in range(len(self.estimators_)):
                    predict_code_matrix[:, n] = _predict_binary(self.estimators_[n], X)

    def compute_loss(self, Y, validate_y, soft_code_matrix):
        """compute the loss"""
        classes_index = dict((c, i) for i, c in enumerate(self.classes_))
        distance_sample_matrix = 0
        for i in range(validate_y.shape[0]):
            distance_sample_matrix += abs(self.decoder._distance
                                          (Y[i], soft_code_matrix[classes_index[validate_y[i]]]))
        distance_sample_matrix /= validate_y.shape[0]
        mindis = 65536
        for i in range(soft_code_matrix.shape[0]):
            for j in range(i):
                distanceij = self.decoder._distance(soft_code_matrix[i], soft_code_matrix[j])
                if distanceij < mindis:
                    mindis = distanceij
        effect = True
        for j in range(soft_code_matrix.shape[1]):
            unique_y = np.unique(soft_code_matrix[:, j])
            if len(unique_y) == 1:
                effect = False
                break

        return distance_sample_matrix, mindis, effect

class SimpleECOCClassifier2(SimpleECOCClassifier):

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        Y = np.array([_predict_binary2(self.estimators_[i], X) for i in range(len(self.estimators_))]).T

        # Y = np.array([self.scaler_[i].transform(Y[:, i]) for i in range(len(self.scaler_))])
        # Y_min, Y_max = Y.min(), Y.max()
        # print('%s: (%f , %f)' % (self.estimator_type, Y_min, Y_max))

        # if self.estimator_type == 'decision_function':
        # Y = _min_max_normalize(Y)  # Use a normalization because scale of Y is [-1,1]

        # Y = Y * 2 - 1  # mapping scale [0, +1] to [-1, +1]

        # Y = np.tanh(Y)

        distances = self.decoder.decode(Y, self.code_matrix)

        pred = distances.argmin(axis=1)


        return self.classes_[pred], distances, Y

# def compute_soft_code_matrix(Y, y, y_index, code_matrix):
#     """compute soft code matrix
#         Parameters
#         Y: vector the classifiers predict
#         y: the label of samples"""
#     df = pd.DataFrame(Y)
#     soft_code_matrix = df.groupby(y).mean()
#     # print(soft_code_matrix)
#     return np.array(soft_code_matrix)

def compute_soft_code_matrix(Y, y, code_matrix):
    """compute soft code matrix
        Parameters
        Y: vector the classifiers predict
        y: the label of samples"""
    df = pd.DataFrame(Y)
    soft_code_matrix = df.groupby(y).mean()
    # print(soft_code_matrix)
    return np.array(soft_code_matrix)



def compute_soft_code_matrix(Y, y, y_index, code_matrix):
    """compute soft code matrix
        Parameters
        Y: vector the classifiers predict
        y: the label of samples"""
    df = pd.DataFrame(Y)
    soft_code_matrix = np.array(df.groupby(y).mean())
    labels = sorted(np.unique(y_index))
    new_matrix = copy.deepcopy(code_matrix)
    for i in range(len(labels)):
        new_matrix[labels[i],:] = soft_code_matrix[i, :]
    return new_matrix



