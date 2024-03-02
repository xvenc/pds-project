"""
log-monitor.py
PDS Project - Log file analysis and anomaly detection
Author: Václav Korvas (xkorva03)
Main module for log monitoring and anomaly detection
"""

import pandas as pd
import argparse
import matplotlib.pyplot as plt
from src.dataload import DataLoader
from src.preprocess import Preprocess
from src.model import ML_model
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier

def accuracy_graph(acurracy_dict):
    """
    Plot the accuracy scores for multiple models
    """
    plt.figure(figsize=(12, 8))
    plt.bar(acurracy_dict.keys(), acurracy_dict.values(), color='blue')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.show()

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Log monitoring and anomaly detection tool')
    parser.add_argument('--log_file', type=str, help='Path to the log file')
    parser.add_argument('--template', type=str, help='Path to the log template file')
    parser.add_argument('--labels', type=str, help='Path to the labels file')
    parser.add_argument('--model', type=str, help='Model for anomaly detection')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    loader = DataLoader(args.log_file, args.template, args.labels) 
    preprocess = Preprocess()
    df = loader.read_log_csv('./logs/hdfs.csv')


    df = preprocess.preprocess(df)
    df = preprocess.normal_anomaly_ratio(df, 0.9)
    preprocess.statistics(df)
    train_data, test_data, train_labels, test_labels = preprocess.data_split(df, 0.25)

    if args.model == 'isolation_forest':
        model = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0)
    elif args.model == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

    ml_model = ML_model(model)

    ml_model.train(train_data, train_labels)
    y_pred = ml_model.predict(test_data)

    acc, f1, prec, conf_matrix = ml_model.evaluate(test_data, test_labels, show=True)
