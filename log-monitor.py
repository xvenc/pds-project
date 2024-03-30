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

    return parser.parse_args()

def perform_experiments(df, model):
    """
    Perform experiments with the model
    """
    pass

if __name__ == '__main__':
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
    train_data, test_data, train_labels, test_labels = preprocess.data_split(df_balanced, 0.25, True)
    print("\nUnbalanced dataset")
    preprocess.statistics(df_unbalanced)
    train_data_unbalanced, test_data_unbalanced, train_labels_unbalanced, test_labels_unbalanced = preprocess.data_split(df_unbalanced, 0.25, True)

    if args.model == 'isolation_forest':
        model = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0)
    elif args.model == 'random_forest':
        model = RandomForestClassifier(criterion='gini', n_estimators=50, min_samples_split=2, max_features='log2', max_depth=20)

    ml_model = ML_model(model)
    ml_model_unbalanced = ML_model(model)
    ml_model_def = ML_model(RandomForestClassifier())
    ml_model_def_unbalanced = ML_model(RandomForestClassifier())

    #params = {'n_estimators': [10, 50, 100, 200], 
    #          'criterion' : ['gini', 'entropy'],
    #          'max_depth' : [2, 6, 10, 20],
    #          'min_samples_split' : [2, 4, 8],
    #          'max_features' : ['sqrt', 'log2'],
    #        }

    #best = ml_model.find_parameters(train_data, train_labels, params)
    #print(best)

    print("\nDefault model balanced:")
    ml_model_def.train(train_data, train_labels)
    y_pred_def = ml_model_def.predict(test_data)

    _ = ml_model_def.evaluate(test_data, test_labels, show=True)

    print("\nDefault model unbalanced:")
    ml_model_def_unbalanced.train(train_data_unbalanced, train_labels_unbalanced)
    y_pred_def_unbalanced = ml_model_def_unbalanced.predict(test_data_unbalanced)

    _ = ml_model_def_unbalanced.evaluate(test_data_unbalanced, test_labels_unbalanced, show=True)


    ml_model.train(train_data, train_labels)
    y_pred = ml_model.predict(test_data)

    print("\nBalanced dataset tuned:")
    acc, f1, prec, rec, conf_matrix = ml_model.evaluate(test_data, test_labels, show=True)

    ml_model_unbalanced.train(train_data_unbalanced, train_labels_unbalanced)
    y_pred_unbalanced = ml_model_unbalanced.predict(test_data_unbalanced)

    print("\nUnbalanced dataset:")
    acc, f1, prec, rec, conf_matrix = ml_model_unbalanced.evaluate(test_data_unbalanced, test_labels_unbalanced, show=True)


    #ml_model.conf_matrix_graph(conf_matrix)