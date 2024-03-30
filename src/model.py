"""
model.py
PDS Project - Log file analysis and anomaly detection
Author: Václav Korvas (xkorva03)
Module with machine learning model for anomaly detection
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

class ML_model:

    def __init__(self, model):
        self.model = model

    def train(self, train_data, train_labels):
        """
        Train the model
        """
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        """
        Predict the labels for the input data
        """
        return self.model.predict(test_data)


    def evaluate(self, X, y_true, show=False):
        """
        Evaluate the model
        """
        y_pred = self.model.predict(X)

        acc = self.accuracy(y_true, y_pred)
        f1 = self.f1(y_true, y_pred)
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        conf_matrix = self.confusion_matrix(y_true, y_pred)

        if show:
            print("Accuracy: ", round(acc, 4))
            print("F1: ", round(f1, 4))
            print("Precision: ", round(prec, 4))
            print("Recall: ", round(rec, 4))
            print("FPR: ", round(conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[0][0]), 4))
            print("Confusion matrix: \n", conf_matrix)

        return acc, f1, prec, rec, conf_matrix

    def find_parameters(self, train_data, train_labels, param_grid):
        """
        Find the best parameters for the model based on the input grid
        """
        grid_search = GridSearchCV(self.model, param_grid, cv=10, scoring='f1', verbose=2)
        grid_search.fit(train_data, train_labels)

        return grid_search.best_params_

    def confusion_matrix(self, y_true, y_pred):
        """
        Return the confusion matrix for the input labels and predictions
        """
        return confusion_matrix(y_true, y_pred)

    def accuracy(self, y_true, y_pred):
        """
        Return the accuracy score for the input labels and predictions
        """
        return accuracy_score(y_true, y_pred)

    def f1(self, y_true, y_pred):
        """
        Return the F1 score for the input labels and predictions
        """
        return f1_score(y_true, y_pred)

    def recall(self, y_true, y_pred):
        """
        Return the recall score for the input labels and predictions
        """
        return recall_score(y_true, y_pred)

    def precision(self, y_true, y_pred):
        """
        Return the precision score for the input labels and predictions
        """
        return precision_score(y_true, y_pred)

    def conf_matrix_graph(self, conf_matrix):
        """
        Plot the confusion matrix as a heatmap
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues_r')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

    def feature_importance(self, df : pd.DataFrame):
        """
        Plot the feature importance of the model
        """

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title("Feature importances")
        plt.bar(range(df.shape[1]), importances[indices], align="center")
        plt.xticks(range(df.shape[1]), indices)
        plt.xlim([-1, df.shape[1]])
        plt.show()

    def save_results(self, df : pd.DataFrame, file_name, f_path):
        """
        Save the results to a csv file
        """
        if not os.path.exists(f_path):
            os.makedirs(f_path)

        df.to_csv(f_path + file_name, index=False)
    
    def create_results(self, models, results):
        """
        Create a dataframe with the results
        """
        cols = ['Model', 'Accuracy', 'F1', 'Precision', 'Recall']
        return results