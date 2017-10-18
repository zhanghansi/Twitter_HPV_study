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

# fieldnames = ['id', 'text', 'clean_text', 'place', 'user_location', 'us_state', 'created_at', 'username', 'user_id','sentiment']
# fieldnames = ['id', 'text', 'clean_text', 'place', 'user_location', 'us_state', 'created_at', 'username', 'user_id','classifier','topic']
fieldnames = ['us_state', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29']
filenames_topic = ['nj', 'tx', 'vt', 'fl', 'dc', 'mn', 'md', 'il', 'in', 'ne', 'or', 'mi', 'al', 'tn', 'ms', 'hi', 'pa', 'me', 'co', 'az', 'id', 'ut', 'nv', 'wi', 'nh', 'ri', 'oh', 'sd', 'ia', 'ny', 'ky', 'ga', 'sc', 'ks', 'wy', 'ok', 'la', 'nc', 'ct', 'ca', 'ar', 'nm', 'wv', 'mt', 'wa', 'va', 'ma', 'nd', 'de', 'ak', 'mo']

def to_csv(us_states, csv_output_file):

    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
        
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)

        writer.writeheader()
        
        for us_state in us_states:
            row = {}
            for field in fieldnames:
                row[field] = '';
            row['us_state'] = us_state;
            cnt = 0
            for n in us_states[us_state]:
                row[str(cnt)] = n
                cnt += 1
            # logger.info(row)
            writer.writerow(row)

def to_csv_normal(tweets, csv_output_file):
        with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
            writer = csv.DictWriter(csv_f, fieldnames=filenames_topic, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for tweet in tweets:

                writer.writerow(tweet)

def state_topic_stats(path):
    with open(path, 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        topic_stats = {}
        # cnt = 0
        for row in reader:
            topic_set = row['topic'][1:len(row['topic'])-1].replace(' ', '').split(',')
            if topic_set[0] == '':
                continue
            if row['us_state'] not in topic_stats:
                topic_stats[row['us_state']] = [0]*30
                for ids in topic_set:
                    topic_stats[row['us_state']][int(ids)] = 1
            else:
                for ids in topic_set:
                    topic_stats[row['us_state']][int(ids)] += 1
            # if row['us_state'] == 'az':
            #     if '0' in topic_set:
            #         cnt += 1

        # logger.info(topic_stats)
        to_csv(topic_stats, './all_data/senti_all/topics_stats/all_topics/0.35/topic_state_level.csv')


def topic_distribution(path, topic_id):
    with open(path, 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        topic_stats = {}
        for row in reader:
            topic_set = row['topic'][1:len(row['topic'])-1].replace(' ', '').split(',')
            if row['us_state'] not in topic_stats:
                if str(topic_id) in topic_set:
                    topic_stats[row['us_state']] = 1
            else:
                if str(topic_id) in topic_set:
                    topic_stats[row['us_state']] += 1
        return topic_stats

def all_topics_stats(path):
    topic_set = range(30)
    resutls = []
    for topic_id in topic_set:
        topic_stats = topic_distribution(path,topic_id)
        resutls.append(topic_stats)
    to_csv_normal(resutls,'./all_data/senti_all/topic_distribution.csv')

def find_top_three(data,position,total):
    test = []
    for n in position:
        test.append(data[n])
    # logger.info(test)
    for i in range(5):
        maximun = max(test)
        # logger.info(maximun)
        for j in position:
            if data[j] == maximun:
                logger.info(j)
        logger.info(maximun/total)
        test.remove(maximun)



def popular_topic_stats(path,state):
    with open(path, 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        topic_stats = {}
        # cnt = 0
        for row in reader:
            topic_set = row['topic'][1:len(row['topic'])-1].replace(' ', '').split(',')
            if row['us_state'] not in topic_stats:
                topic_stats[row['us_state']] = [0]*30
                for ids in topic_set:
                    topic_stats[row['us_state']][int(ids)] = 1
            else:
                for ids in topic_set:
                    topic_stats[row['us_state']][int(ids)] += 1
        Posi = [4,9,12,15,16,17,18,20,21,22,25]
        Neg = [5,6,10,13,19,29]
        total = 0
        for n in topic_stats[state]:
            total += n
        logger.info(total)
        # posi_top = find_top_three(topic_stats[state],Posi,total)
        # neg_top = find_top_three(topic_stats[state],Neg,total)
        top_3 = find_top_three(topic_stats[state],range(30),total)

if __name__ == "__main__":

    logger.info(sys.version)
    # popular_topic_stats('./all_data/senti_all/senti_topics_0.25.csv', 'ri')
    state_topic_stats('./intermediate_data/promotional/cutoffline/promotional_topics_0.25.csv')
    # all_topics_stats('./all_data/senti_all/senti_topics_0.25.csv')
    # logger.info(topic_distribution('./all_data/ad_all/ad_topics_0.2.csv',5))