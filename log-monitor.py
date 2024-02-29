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
    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('log_template', type=str, help='Path to the log template file')
    parser.add_argument('labels', type=str, help='Path to the labels file')
    parser.add_argument('model', type=str, help='Model for anomaly detection')
    return parser.parse_args

if __name__ == '__main__':
    args = parse_args()