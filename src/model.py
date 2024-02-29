"""
model.py
PDS Project - Log file analysis and anomaly detection
Author: Václav Korvas (xkorva03)
Module with machine learning model for anomaly detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
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


    def evaluate(self, y_test, model_name):
        """
        Evaluate the model
        """

        y_test_pred = self.model.predict(y_test)

        print('Evaluation of the', model_name, 'model')
        print('Accuracy:', accuracy_score(y_test, y_test_pred))
        print('F1:', f1_score(y_test, y_test_pred))
        print('Precision:', precision_score(y_test, y_test_pred))
        print('Confusion matrix:')
        print(confusion_matrix(y_test, y_test_pred))


    def find_parameters(self, train_data, train_labels, param_grid):
        """
        Find the best parameters for the model based on the input grid
        """
        grid_search = GridSearchCV(self.model, param_grid, cv=10, scoring='f1', verbose=2)
        grid_search.fit(train_data, train_labels)

        return grid_search.best_params_
    
    def confusion_matrix(self, test_labels, predictions):
        """
        Return the confusion matrix for the input labels and predictions
        """
        return confusion_matrix(test_labels, predictions)

    def accuracy(self, test_labels, predictions):
        """
        Return the accuracy score for the input labels and predictions
        """
        return accuracy_score(test_labels, predictions)
    
    def f1(self, test_labels, predictions):
        """
        Return the F1 score for the input labels and predictions
        """
        return f1_score(test_labels, predictions)
    
    def precision(self, test_labels, predictions):
        """
        Return the precision score for the input labels and predictions
        """
        return precision_score(test_labels, predictions)
    
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