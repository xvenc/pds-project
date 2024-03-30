"""
log-monitor.py
PDS Project - Log file analysis and anomaly detection
Author: Václav Korvas (xkorva03)
Main module for log monitoring and anomaly detection
"""

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

def perform_experiments(df, models, model_names, balanced_data, unbalanced_data):
    """
    Perform experiments with the models
    """
    i = 0
    for model, name in zip(models, model_names):
        print("\nModel: ", name)
        if i % 2 == 0:
            model.train(balanced_data['train_data'], balanced_data['train_labels'])
            y_pred = model.predict(balanced_data['test_data'])
            acc, f1, prec, rec, conf_matrix = model.evaluate(balanced_data['test_data'], balanced_data['test_labels'], show=True)
        else:
            model.train(unbalanced_data['train_data'], unbalanced_data['train_labels'])
            y_pred = model.predict(unbalanced_data['test_data'])
            acc, f1, prec, rec, conf_matrix = model.evaluate(unbalanced_data['test_data'], unbalanced_data['test_labels'], show=True)    

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
