#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s]: %(levelname)s: %(message)s')

import os
import re
import json
import sys
import common
import csv
import json

import operator
## 1 : ad ; 2 : people ;

def classify(csv_file):
    tweets = []
    cnt_ad = 0
    cnt_laypeople = 0

    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        for tweet in reader:
            flag = 0
            text = tweet['text'].lower()

            if ('http' in text or text.endswith('htt') or text.endswith('ht')):
                tweet['class'] = '1'
                cnt_ad += 1
            else:
                tweet['class'] = '2'
                cnt_laypeople += 1

            tweets.append(tweet)

    logger.info('laypeople: %d; ad: %d'%(cnt_laypeople,cnt_ad))
    return tweets

fieldnames = ['id', 'text', 'clean_text', 'place', 'user_location', 'us_state', 'created_at', 'username', 'user_id','class']
def to_csv(tweets, csv_output_file):
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
        
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)

        writer.writeheader()
        
        for tweet in tweets:
            writer.writerow(tweet)


def extract_tweets_by_label(label, input_file,csv_output_file):
    tweets = []
    with open(input_file, 'r', newline='', encoding='utf-8', errors='ignore') as csv_f:
        reader = csv.DictReader(csv_f)
        for tweet in reader:
            if( tweet['class'] == label):
                tweets.append(tweet)
            else:
                continue
    to_csv(tweets, csv_output_file)


if __name__ == "__main__":

    logger.info(sys.version)

    tweets = classify('./intermediate_data/hpv_geotagged.csv')
    to_csv(tweets, './intermediate_data/classified_hpv.csv')


    extract_tweets_by_label('1', './intermediate_data/classified_hpv.csv','./intermediate_data/promotional/promotional.csv')
    extract_tweets_by_label('2', './intermediate_data/classified_hpv.csv','./intermediate_data/laypeople/laypeople.csv')