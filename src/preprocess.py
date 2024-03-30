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

    def preprocess(self, df : pd.DataFrame, cols) -> pd.DataFrame:
        """
        Preprocesses the input dataframe into a format suitable for machine learning
        """

        columns_to_drop = ['Date', 'Time', 'PID', 'Content', 'Level', 'Component']

        # Prepend the E0 column to the beginning of the list
        cols = ['E0'] + cols
        sorted_cols = sorted(cols, key=lambda x: int(x[1:]))
        labels = df[['Label', 'BlockId']].drop_duplicates()
        df = df.drop(columns=columns_to_drop)
        df = df.groupby(['BlockId', 'EventId']).size().unstack(fill_value=0).reset_index()
        missing_columns = [col for col in sorted_cols if col not in df.columns]
        for col in missing_columns:
            df[col] = 0
        df = df[['BlockId'] + sorted_cols]
        df = pd.merge(df, labels, on='BlockId')

        # Normalization
        min_max = min_max = df.iloc[:, 1:-1].agg([np.min, np.max])
        global_min = min_max.loc['amin'].min()
        global_max = min_max.loc['amax'].max()
        df.iloc[:, 1:-1] = (df.iloc[:, 1:-1] - global_min) / (global_max - global_min)

        df['Label'] = df['Label'].map({'Normal': 0, 'Anomaly': 1})
        df = df.drop(columns=['BlockId'])

        return df

    def normal_anomaly_ratio(self, df : pd.DataFrame, normal_ratio):
        """
        Balance the input dataframe by dropping some normal samples
        """

        anomaly_cnt = len(df[df['Label'] == 1])
        desired_normal_cnt = int((anomaly_cnt/ (1-normal_ratio)) - anomaly_cnt)

        # Shuffle the dataframe
        df = df.sample(frac=1).reset_index(drop=True)

        normal = df[df['Label'] == 0]
        normal = normal.sample(desired_normal_cnt)
        df = pd.concat([normal, df[df['Label'] == 1]])

        return df

    def data_split(self, df : pd.DataFrame, test_size, show=False):
        """
        Split the input dataframe into train and test sets
        """
        labels = df['Label']
        df = df.drop(columns=['Label'])
        matrix = np.array(df.values.tolist())
        train_data, test_data, train_labels, test_labels = train_test_split(matrix, labels, test_size=test_size, random_state=42)

        if (show):
            print("Train normal: ", len(train_labels[train_labels == 0]))
            print("Train anomaly: ", len(train_labels[train_labels == 1]))
            print("Test normal: ", len(test_labels[test_labels == 0]))
            print("Test anomaly: ", len(test_labels[test_labels == 1]))

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