"""
dataload.py
PDS Project - Log file analysis and anomaly detection
Author: Václav Korvas (xkorva03)
Module with data loading
"""

import re
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:

    processed = 0
    unknown_label = 0
    matched = 0
    unknown_event = 0
    skipped = 0

    def __init__(self, log_template_file="./logs/HDFS_templates2.csv", label_file="./logs/anomaly_label.csv"):
        self.log_template_file = log_template_file
        self.label_file = label_file

    def read_log(self, log_file ,split=False, labels=False):
        """
        Read the log file line by line into a list and strip trailing whitespace characters
        """
        with open(log_file, 'r') as f:
            print("Reading log file...")
            log = f.readlines()
            log = [l.strip() for l in log]

        # Split the log line into separate columns
        if split and not labels:
            log = [l.split(" ", 5) for l in log]
        elif split and labels:
            # Log entries with labels have the label at the end
            log = [l.split(" ", 5) for l in log]
            # Now the last column split on the last space
            log = [[*l[:-1], *l[-1].rsplit(" ", 1)] for l in log]

        return log

    def read_log_template_csv(self):
        """
        Read the csv log template file into a list and return it without the header
        """
        with open(self.log_template_file, 'r') as f:
            print("Reading log template file...")
            log_template = list(csv.reader(f))

        return self._replace_string_template(log_template[1:])

    def read_label_csv(self):
        """
        Read the csv label file into a dictionary and return it
        """
        data = {}
        with open(self.label_file, 'r') as f:
            print("Reading label file...")
            for row in csv.reader(f):
                data[row[0]] = row[1]
            data.pop('BlockId')
        return data

    def read_log_csv(self, file_name):
        """
        Load the log from a csv file into a pandas dataframe
        """
        print("Reading log file...")
        df = pd.read_csv(file_name)
        return df
    
    def _replace_string_template(self, log_template):
        """
        Replace special characters in the log template, so they can be used in regex matching
        """
        for line in log_template:
            line[1] = line[1].replace('BLOCK*', 'BLOCK\\*')
            line[1] = line[1].replace('<*>', '.*')
        
        return log_template

    def _extract_blk_id(self, line):
        """
        Extract the block id from a log line
        """
        blk_id = re.findall(r'blk_-?\d+', line)
        if len(blk_id) == 0:
            return None
        return blk_id[0]

    def _extract_label(self, anotations, blk_id):
        """
        Extract the label based on the block id
        """
        if blk_id in anotations:
            return anotations[blk_id]
        else:
            return None

    def extract_template_events(self):
        """
        Extract all possible events from the log template file
        """
        with open(self.log_template_file, 'r') as f:
            log_template = list(csv.reader(f))
            event_id = [pat[0] for pat in log_template[1:]]
        
        return event_id

    def to_dataframe(self, events):
        """
        Convert the list of events into a pandas dataframe
        """
        columns = ['Date', 'Time', 'PID', 'Level', 'Component', 'Content', 'EventId', 'BlockId', 'Label']
        df = pd.DataFrame(events, columns=columns)
        return df

    def df_to_csv(self, df, file_name):
        """
        Save the dataframe to a csv file
        """
        df.to_csv(file_name, index=False)

    def split_logfile(self, log, training_size):
        """
        Split the log file dataframe into training and testing data
        """
        train, test = train_test_split(log, test_size=training_size)
        return train, test

    def convert_df_to_list(self, df):
        """
        Convert the dataframe to a list
        """
        return df.values.tolist()

    def save_logfile(self, log, file_name, file_path):
        """
        Save the log file to other file
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        with open(file_path + file_name, 'w') as f:
            for l in log:
                line = ' '.join(l)
                f.write(line + '\n')

    def match_log_label(self, log, labels):
        """
        Match the log with the labels
        """
        for l in log:
            blk_id = self._extract_blk_id(l[-1])
            label = self._extract_label(labels, blk_id)
            if label:
                l.append(label)
            else:
                l.append('Anomaly')
        return log

    def match_event(self, log, log_template, labels, labels_present=False):
        """
        Match the events in the log with the log template and labels
        """
        print("Parsing log...")
        for l in log:
            if labels_present:
                content = l[-2]
                label = l.pop()
            else:
                content = l[-1]
            blk_id = self._extract_blk_id(content)
            if not labels_present:
                label = self._extract_label(labels, blk_id)
            event_id = None
            for pat in log_template:
                pattern = re.compile(pat[1])
                match = pattern.match(content)
                if match:
                    event_id = pat[0]
                    break
            
            if blk_id and event_id and label:
                self.matched += 1
                event = [event_id, blk_id, label]
                l.extend(event)
            elif blk_id and label and not event_id:
                # If the event id is not present, mark the event as E0
                event = ['E0', blk_id, label]
                l.extend(event)
                self.unknown_event += 1
            elif blk_id and not label and event_id:
                # If the label is not present, mark the event as an anomaly
                event = [event_id, blk_id, 'Anomaly']
                l.extend(event)
                self.unknown_label += 1
            else:
                # Skip the line if it doesn't contain the block id
                self.skipped += 1

            self.processed += 1

        return log

    def get_stats(self):
        """
        Get statistics about the matching process
        """
        print("Processed: ", self.processed)
        print("Matched: ", self.matched)
        print("Unknown label: ", self.unknown_label)
        print("Unknown event: ", self.unknown_event)
        print("Skipped: ", self.skipped)