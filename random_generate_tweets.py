#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


import os
import sys
import math
import csv
import common
import random

fieldnames = ['id', 'text', 'clean_text', 'place', 'user_location', 'us_state', 'created_at', 'username', 'user_id','class']
# fieldnames = ['id','label','raw_text', 'text']
def to_csv(tweets, csv_output_file):
        with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for tweet in tweets:

                writer.writerow(tweet)

if __name__ == "__main__":

    logger.info(sys.version)
    samples_number = random.sample(range(1, 272094), 100)
    logger.info(samples_number)
    sample_data = []
    result = []
    with open('./intermediate_data/classified_hpv.csv', 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        for row in reader:
            # if 'health' in row['text'].lower():
                # if 'lead' in row['text'].lower():
                #     if 'deadly' in row['text'].lower():
                #          logger.info(row['text'])
            sample_data.append(row)

    for i in samples_number:
        result.append(sample_data[i])

    # fieldnames = ['id', 'text', 'clean_text', 'place', 'user_location', 'us_state', 'created_at', 'username', 'user_id','classifier']
    to_csv(result,'./intermediate_data/analysis/annotation_for_rule_based_categorization/random_100.csv')
