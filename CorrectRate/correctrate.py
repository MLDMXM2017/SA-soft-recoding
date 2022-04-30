import numpy as np
import pandas as pd


class Indicator:
    @staticmethod
    def get_indicators(predict_value, actual_value):
        """compute accuracy,precision,recall,f1-score
        Parameters:
            predict_value:the labels the classifiers predict
            actual_value:the actual labels"""
        acc_pre_rec_f1 = []
        count = 0
        for i in range(predict_value.shape[0]):
            if predict_value[i] == actual_value[i]:
                count += 1
        acc_pre_rec_f1.append((count/predict_value.shape[0]).__round__(4))
        group_value = pd.DataFrame(predict_value).groupby(predict_value)
        precision = []
        recall = []
        label, times = np.unique(actual_value, return_counts=True)
        frequency = dict(zip(label, times))
        for i in group_value:
            TP = 0
            FP = 0
            for j in range(i[1].shape[0]):
                if i[1].iloc[j].values[0] == actual_value[i[1].iloc[j].name]:
                    TP += 1
                else:
                    FP += 1
            precision.append(TP/(TP+FP))
            if TP == 0:
                recall.append(0)
            else:
                recall.append(TP/frequency.get(i[0]))
        acc_pre_rec_f1.append(np.mean(precision).__round__(4))
        acc_pre_rec_f1.append(np.mean(recall).__round__(4))
        acc_pre_rec_f1.append((2*(np.mean(precision)*np.mean(recall))
                              / (np.mean(precision)+np.mean(recall))).__round__(4))
        return acc_pre_rec_f1

    @staticmethod
    def get_confusion_matrix(predict_value, actual_value):
        """get confusion matrix
        Parameters:
            predict_value:the labels the classifiers predict
            actual_value:the actual labels"""
        all_class = np.unique(actual_value)
        confusion_matrix = pd.DataFrame(index=all_class, columns=all_class)
        connect = np.concatenate((np.array([predict_value]).T, np.array([actual_value]).T), axis=1)
        for i_class in all_class:
            target = predict_value[connect[:, -1] == i_class]
            #print(i_class)
            #print(confusion_matrix)
            #print(pd.DataFrame(target).groupby(target).count())
            i_class_target = pd.DataFrame(target).groupby(target).count()
            #print(i_class_target.index.values)
            for each_index in i_class_target.index.values:
                confusion_matrix.loc[i_class, each_index] = i_class_target.loc[each_index, 0]
        confusion_matrix.fillna(0)
        return np.array(confusion_matrix)

    @staticmethod
    def get_distances_change(old_distances, new_distances, labels):
        columns_index = list(range(1, old_distances.shape[1]+1))
        columns_index.append(" ")
        columns_index.append("label")
        columns_index.append(" ")
        columns_index.extend(list(range(1, old_distances.shape[1]+1)))
        empty_list = [""] * old_distances.shape[0]
        data = np.concatenate([old_distances, np.vstack([empty_list, labels, empty_list]).T, new_distances], axis=1)
        table = pd.DataFrame(data, index=list(range(1, old_distances.shape[0]+1)), 
        columns=columns_index)
        return table.sort_values(by='label')


















