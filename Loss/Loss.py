""" get loss"""


import numpy as np
from scipy.spatial.distance import cdist
import copy


class Loss:

    def __init__(self, classes, code_matrix):
        self.classes_ = classes
        self.code_matrix = code_matrix
        self.classes_index = dict((c, i) for i, c in enumerate(self.classes_))

    def update_code_matrix_by_row(self, Y, y, sec, alpha=0.1, T=500, ε=0.00001):

        iteration_times = T
        y = np.array(y)
        self.label_to_index = np.array([self.classes_index[l] for l in y])
        b = np.zeros((self.code_matrix.shape[0], self.code_matrix.shape[1]))
        w = np.ones((self.code_matrix.shape[0], self.code_matrix.shape[1]))
        origin_codematrix = copy.deepcopy(self.code_matrix)
        for current_time in range(iteration_times):
            initial_code_matrix = copy.deepcopy(self.code_matrix)
            origin_loss, target_distances, other_distances, total_distances = self.get_loss_by_row(Y, y)
            for i in self.classes_:
                row = self.classes_index[i]
                for j in range(self.code_matrix.shape[1]):

                    loss, target_distances, other_distances, total_distances = self.get_loss_by_row(Y, y)
                    db = self.get_gradient_by_row(row, j, Y, y, total_distances)
                    dw = db * origin_codematrix[row, j]
                    b[row, j] = b[row, j] - alpha * db
                    w[row, j] = w[row, j] - alpha * dw
                    old_value = self.code_matrix[row, j]
                    tmp = self.code_matrix[row, j] - alpha * (origin_codematrix[row, j] * dw + db)
                    if tmp > 1:
                        tmp = old_value
                    elif tmp < -1:
                        tmp = old_value
                    self.code_matrix[row, j] = tmp

            update_loss, update_target_distances, update_other_distances, update_total_distances = self.get_loss_by_row(Y, y)
            sec.set_code_matrix(self.code_matrix)
            if origin_loss < update_loss:
                alpha = alpha * 0.95
                self.code_matrix = initial_code_matrix
            elif np.abs(origin_loss - update_loss) < ε:
                break
        return self.code_matrix
    
    @staticmethod
    def get_distance(Y, M):
        return cdist(Y, M)

    def get_loss_by_row(self, Y, y):
        total_distances = self.get_distance(Y, self.code_matrix)
        target_distances = np.choose(self.label_to_index, total_distances.T)
        other_distances = np.sum(total_distances, axis=1) - target_distances + 0.00001
        loss = np.sum(target_distances/other_distances)
        return loss, target_distances, other_distances, total_distances

    def get_gradient_by_row(self, row, j, Y, y, total_distances):
        inner_distances = np.choose(self.label_to_index, total_distances.T)
        row_samples_distances = total_distances[self.label_to_index == row, :]
        sample_inner_distances = inner_distances[self.label_to_index == row] + 0.00001
        sample_outer_distances = np.sum(row_samples_distances, axis=1) - sample_inner_distances + 0.00001
        loss1 = np.sum((Y[self.label_to_index == row, j] - self.code_matrix[row, j]) / (sample_inner_distances * sample_outer_distances))
        other_samples_distances = total_distances[self.label_to_index != row, :]
        other_inner_distance = inner_distances[self.label_to_index != row] + 0.00001
        other_outer_distance = np.sum(other_samples_distances, axis=1) - other_inner_distance + 0.00001
        loss2 = -np.sum((other_inner_distance * (Y[self.label_to_index != row, j] - self.code_matrix[row, j]))
                        / (other_samples_distances[:, row] * np.power(other_outer_distance, 2)))
        db = -(loss1 + loss2) * (self.code_matrix.shape[0] - 1)
        return db

