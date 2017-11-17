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
import time
import pandas as pd
import ftfy
import common
import csv

import pprint
pp = pprint.PrettyPrinter(indent=4)


import nltk
from nltk.stem import WordNetLemmatizer
pattern=r'[A-Z]\.+\S+|\w+\-\w+|\w+'

def load_stoplist():
    stoplist = set()
    with open('./english.stop.txt', newline='', encoding='utf-8') as f:
        for line in f:
            stoplist.add(line.strip())
    return stoplist


def extract_clean_text(json_file):
    wv = []
    cnt = 0
    stoplist = load_stoplist()
    wordnet_lemmatizer = WordNetLemmatizer()
    with open(json_file, 'r') as json_file:
        user_tweets = json.load(json_file)
        for user in user_tweets:
            text = ''
            for tweet in user_tweets[user]:
                text += common.cleanhtml(common.remove_hashtag_sign(common.remove_username(common.remove_url(ftfy.fix_text(tweet))))) + ' '
            clean_texts = [wordnet_lemmatizer.lemmatize(word.lower()) for word in nltk.regexp_tokenize(text, pattern) if wordnet_lemmatizer.lemmatize(word.lower()) not in stoplist]
            wv.append(clean_texts)
            cnt += 1
    logger.info('total tweets: %d;'%cnt)
    return wv

def to_txt(wv, output_file):
    with open(output_file, 'w') as txt_file:
        for tweet in wv:
            for word in tweet:
                txt_file.write(word + ' ')
            txt_file.write('\n')

def extract_tweet_not_by_uid(source,output_file):
    tweets = []
    df = pd.read_csv(source,encoding='utf-8')
    for index, row in df.iterrows():
        text = common.cleanhtml(common.remove_username(common.remove_url(ftfy.fix_text(row['text']))))
        tags = re.findall(r"#(\w+)", text)
        if len(tags) != 0:
            tweets.append(text)
    with open(output_file, 'w') as outfile:
        json.dump(tweets, outfile)
    print(len(tweets))
if __name__ == "__main__":

    logger.info(sys.version)

    extract_tweet_not_by_uid('./intermediate_data/hpv_geotagged.csv','./intermediate_data/hpv_tweets/hpv_tweets_not_by_uid.txt')


    # wv = extract_clean_text('./intermediate_data/userid_list.json')
    # to_txt(wv, './intermediate_data/hpv_tweets/hpv_tweets.txt')