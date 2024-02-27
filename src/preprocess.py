import re
import csv
import argparse
import pandas as pd

log_file = '../logs/HDFS-test.log'

def parse_args():
    parser = argparse.ArgumentParser("Preprocess the given HDFS log file.")
    parser.add_argument('--log_file', type=str, help='The input log file')
    parser.add_argument('--log_template_file', type=str, help='The input log template file')
    parser.add_argument('--label_file', type=str, help='The input file with labels')
    return parser.parse_args()

def read_log(log_file):
    with open(log_file, 'r') as f:
        log = f.readlines()
        log = [l.strip() for l in log]
    return log

def read_log_template_csv(log_template_file):
    with open(log_template_file, 'r') as f:
        log_template = list(csv.reader(f))
    return log_template[1:]

def read_label_csv(label_file):
    data = {}
    with open(label_file, 'r') as f:
        #label = list(csv.reader(f))
        for row in csv.reader(f):
            data[row[0]] = row[1]
        data.pop('BlockId')
    return data

def to_dataframe(events):
    columns = ['Date', 'Time', 'PID', 'Level', 'Component', 'Content', 'EventId', 'BlockId', 'Label']
    df = pd.DataFrame(events, columns=columns)
    return df

def extract_blk_id(line):
    blk_id = re.findall(r'blk_-?\d+', line)
    if len(blk_id) == 0:
        return None
    return blk_id[0]

def extract_label(anotations, blk_id):
    if blk_id in anotations:
        return anotations[blk_id]
    else:
        return None

def match_event(log, log_template, labels):
    for l in log:
        blk_id = extract_blk_id(l[-1])
        label = extract_label(labels, blk_id)
        event_id = None
        for pat in log_template:
            pattern = re.compile(pat[1])
            match = pattern.match(l[-1])
            if match:
                event_id = pat[0]
                break
        if blk_id and match and label:
            event = [event_id, blk_id, label]
            l.extend(event)
        else:
            print("No match found for: ", l[-1])
    return log 

log = read_log(log_file)
log = [l.split(" ", 5) for l in log]
log_template = read_log_template_csv('../logs/HDFS_templates.csv')
labels = read_label_csv('../logs/anomaly_label.csv')
events = match_event(log, log_template, labels)
df = to_dataframe(events)
print(df.head(10))