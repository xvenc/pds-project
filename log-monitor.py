"""
log-monitor.py
PDS Project - Log file analysis and anomaly detection
Author: Václav Korvas (xkorva03)
Main module for log monitoring and anomaly detection
"""

import argparse
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.dataload import DataLoader
from src.preprocess import Preprocess
from src.model import ML_model
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Log monitoring and anomaly detection tool')
    parser.add_argument('--log_file', type=str, help='Path to the log file')
    parser.add_argument('--template', type=str, help='Path to the log template file')
    parser.add_argument('--labels', type=str, help='Path to the labels file')
    parser.add_argument('--model', type=str, help='Model for anomaly detection')
    parser.add_argument('--parse', type=bool, help='Parse the log file and don''t use the csv files', default=False)
    parser.add_argument('-training', required=True, help='Path to the training data')
    parser.add_argument('-testing', required=True, help='Path to the testing data')

    return parser.parse_args()

def save_results(df : pd.DataFrame, file_name, f_path):
    """
    Save the results to a csv file
    """
    if not os.path.exists(f_path):
        os.makedirs(f_path)

    df.to_csv(f_path + file_name, index=False)

# 0 train, 1 test, 2 train_labels, 3 test_labels
def perform_experiments(models, model_names, data):
    """
    Perform experiments with the models
    """

    result_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall', 'FPR'])
    i = 0
    for model, name in zip(models, model_names):
        print("\nModel: ", name,"\n")
        model.train(data[0], data[2])
        acc, f1, prec, rec, conf_matrix = model.evaluate(data[1], data[3], show=True)
        new_row = {'Model': name, 'Accuracy': acc, 'F1': f1, 'Precision': prec, 'Recall': rec, 'FPR': conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[0][0])}
        result_df.loc[i] = new_row
        print(type(conf_matrix))
        i += 1
    
    return result_df

def old_main():
    args = parse_args()

    loader = DataLoader(args.log_file, args.template, args.labels) 
    preprocess = Preprocess()

    # If the parse argument is set, parse the log file and match the events with the labels
    if args.parse:
        log = loader.read_log(split=True)
        log_template = loader.read_log_template_csv()
        labels = loader.read_label_csv()
        matched_logs = loader.match_event(log, log_template, labels)
        loader.get_stats()
        df = loader.to_dataframe(matched_logs)
    else:
        df = loader.read_log_csv('./logs/hdfs.csv')

    event_ids = loader.extract_template_events()
    df = preprocess.preprocess(df, event_ids)

    df_unbalanced = df.copy()
    df_balanced = preprocess.normal_anomaly_ratio(df, 0.6)

    print("\nBalanced dataset")
    preprocess.statistics(df_balanced)
    balanced_data = preprocess.data_split(df_balanced, 0.25, True)
    print("\nUnbalanced dataset")
    preprocess.statistics(df_unbalanced)
    unbalanced_data = preprocess.data_split(df_unbalanced, 0.25, True)

    if args.model == 'isolation_forest':
        model = IsolationForest()
    elif args.model == 'random_forest':
        model = RandomForestClassifier(criterion='gini', n_estimators=50, min_samples_split=2, max_features='log2', max_depth=20)

    ml_model = ML_model(model)
    ml_model_unbalanced = ML_model(model)
    ml_model_def = ML_model(RandomForestClassifier())
    ml_model_def_unbalanced = ML_model(RandomForestClassifier())

    models = [ml_model, ml_model_unbalanced, ml_model_def, ml_model_def_unbalanced]
    model_names = ['RF tuned balanced', 'RF tuned unbalanced', 'RF default balanced', 'RF default unbalanced']

    #result_df = perform_experiments(models, model_names, balanced_data, unbalanced_data)
    #save_results(result_df, 'results.csv', './out/') 
    #print(result_df)

if __name__ == '__main__':
    
    args = parse_args()

    loader = DataLoader()
    preprocess = Preprocess()

    train = loader.read_log(args.training, split=True, labels=True)
    test = loader.read_log(args.testing, split=True, labels=True)

    log_template = loader.read_log_template_csv()

    matched_train = loader.match_event(train, log_template, None, labels_present=True)
    matched_test = loader.match_event(test, log_template, None, labels_present=True)

    #loader.get_stats()

    event_ids = loader.extract_template_events()

    df_train = loader.to_dataframe(matched_train)
    df_test = loader.to_dataframe(matched_test)

    df_train = preprocess.preprocess(df_train, event_ids)
    df_test = preprocess.preprocess(df_test, event_ids)
    print("\nTrain data")
    preprocess.statistics(df_train)
    print("\nTest data")
    preprocess.statistics(df_test)
    
    train_data, train_labels = preprocess.split_dataframe(df_train)
    test_data, test_labels = preprocess.split_dataframe(df_test)

    model = RandomForestClassifier(criterion='entropy', n_estimators=50, min_samples_split=2, max_features='sqrt', max_depth=20)
    ml_model = ML_model(model)
    ml_model_def = ML_model(RandomForestClassifier())
    ml_model_ISF = ML_model(IsolationForest(contamination=0.3))

    models = [ml_model, ml_model_def]
    model_names = ['Random Forest tuned', 'Random Forest default']

    result_df = perform_experiments(models, model_names, [train_data, test_data, train_labels, test_labels])
    #save_results(result_df, 'results.csv', './out/')

    #ml_model_ISF.train(train_data, train_labels)
    #y_pred = ml_model_ISF.predict(test_data)
    #y_pred = np.where(y_pred > 0, 0, 1)
    #print("\nIsolation Forest")
    #print("Accuracy: ",round(ml_model_ISF.accuracy(test_labels, y_pred),4))
    #print("F1: ",round(ml_model_ISF.f1(test_labels, y_pred),4))
    #print("Precision: ",round(ml_model_ISF.precision(test_labels, y_pred),4))
    #print("Recall: ",round(ml_model_ISF.recall(test_labels, y_pred),4))
    #print("FPR: ", round(ml_model_ISF.fpr(test_labels, y_pred), 4))