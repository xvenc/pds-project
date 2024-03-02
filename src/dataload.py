"""
dataload.py
PDS Project - Log file analysis and anomaly detection
Author: Václav Korvas (xkorva03)
Module with data loading
"""

import re
import csv
import pandas as pd

class DataLoader:

    processed = 0
    skipped = 0
    matched = 0

    def __init__(self, log_file, log_template_file, label_file):
        self.log_file = log_file
        self.log_template_file = log_template_file
        self.label_file = label_file

    def read_log(self, split=False):
        """
        Read the log file line by line into a list and strip trailing whitespace characters
        """
        with open(self.log_file, 'r') as f:
            print("Reading log file...")
            log = f.readlines()
            log = [l.strip() for l in log]

        if split:
            log = [l.split(" ", 5) for l in log]

        return log

    def read_log_template_csv(self):
        """
        Read the csv log template file into a list and return it without the header
        """
        with open(self.log_template_file, 'r') as f:
            print("Reading log template file...")
            log_template = list(csv.reader(f))
        return log_template[1:]

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

    def match_event(self, log, log_template, labels):
        """
        Match the events in the log with the log template and labels
        """
        for l in log:
            blk_id = self._extract_blk_id(l[-1])
            label = self._extract_label(labels, blk_id)
            event_id = None
            for pat in log_template:
                pattern = re.compile(pat[1])
                match = pattern.match(l[-1])
                if match:
                    event_id = pat[0]
                    break
            if blk_id and match and label:
                self.matched += 1
                event = [event_id, blk_id, label]
                l.extend(event)
            else:
                self.skipped += 1
                print("No match found for: ", l[-1])
            self.processed += 1
        return log

    def get_stats(self):
        """
        Get statistics about the matching process
        """
        print("Processed: ", self.processed)
        print("Matched: ", self.matched)
        print("Skipped: ", self.skipped)
