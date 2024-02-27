# TODO preprocess log file to get the id and list of events

import re
import csv
import os
import argparse

log_file = '../logs/HDFS-test.log'

def read_log(log_file):
    with open(log_file, 'r') as f:
        log = f.readlines()
        log = [l.strip() for l in log]
    return log

def read_log_template_csv(log_template_file):
    with open(log_template_file, 'r') as f:
        log_template = list(csv.reader(f))
    return log_template[1:]

def extract_blk_id(log):
    blk_ids = []
    blk_pattern = re.compile(r'blk_-?\d+')
    for l in log:
        blk = blk_pattern.findall(l[-1])
        if blk:
            blk_ids.append(blk[0])
    return blk_ids

def match_event(log, log_template):
    for l in log:
        for pat in log_template:
            pattern = re.compile(pat[1])
            match = pattern.match(l[-1])
            if match:
                print(f"Matched: {l[-1]} with {pat[0]}")
                break
        


log = read_log(log_file)
log_template = read_log_template_csv('../logs/HDFS_templates.csv')
log = [l.split(" ", 5) for l in log]
blk_ids = extract_blk_id(log)
match_event(log, log_template)