"""
preprocess.py
PDS Project - Log file analysis and anomaly detection
Author: Václav Korvas (xkorva03)
Module with data preprocessing and feature extraction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Preprocess:

    def __init__(self):
        pass

    def preprocess(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input dataframe into a format suitable for machine learning
        """

        columns_to_drop = ['Date', 'Time', 'PID', 'Content', 'Level', 'Component']
        sorted_cols = ['E' + str(i) for i in range(1, 30)]
        labels = df[['Label', 'BlockId']].drop_duplicates()

        df = df.drop(columns=columns_to_drop)
        df = df.groupby(['BlockId', 'EventId']).size().unstack(fill_value=0).reset_index()
        df = df[['BlockId'] + sorted_cols]
        df = pd.merge(df, labels, on='BlockId')
        df['Label'] = df['Label'].map({'Normal': 0, 'Anomaly': 1})
        df = df.drop(columns=['BlockId'])

        return df

    def normal_anomaly_ratio(self, df : pd.DataFrame, normal_ratio):
        """
        Balance the input dataframe by dropping some normal samples
        """

        anomaly_cnt = len(df[df['Label'] == 1])
        desired_normal_cnt = int((anomaly_cnt/ (1-normal_ratio)) - anomaly_cnt)

        normal = df[df['Label'] == 0]
        normal = normal.sample(desired_normal_cnt)
        df = pd.concat([normal, df[df['Label'] == 1]])

        return df

    def data_split(self, df : pd.DataFrame, test_size):
        """
        Split the input dataframe into train and test sets
        """
        labels = df['Label']
        df = df.drop(columns=['Label'])
        matrix = np.array(df.values.tolist())
        train_data, test_data, train_labels, test_labels = train_test_split(matrix, labels, test_size=test_size, random_state=42)

        return train_data, test_data, train_labels, test_labels
        

    def statistics(self, df : pd.DataFrame):
        """
        Computes statistics from the input dataframe
        """
        normal = df[df['Label'] == 0]
        abnormal = df[df['Label'] == 1]
        print("Data statistics:")
        print("Total: ", len(df))
        print("Normal: ", len(normal), " percentage: ", round(len(normal) / len(df), 3))
        print("Anomaly: ", len(abnormal), " percentage: ", round(len(abnormal) / len(df),3))