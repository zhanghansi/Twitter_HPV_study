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
import common
import csv
from sortedcontainers import SortedDict

import pprint
pp = pprint.PrettyPrinter(indent=4)


import numpy as np
import matplotlib.pyplot as plt

import random


# return:    {wid:w, ...}
def read_voca(pt):
    voca = {}
    for l in open(pt):
        wid, w = l.strip().split('\t')[:2]
        voca[int(wid)] = w
    return voca

def read_pz(pt):
    return [float(p) for p in open(pt).readline().split()]

# voca = {wid:w,...}
def dispTopics(pt, voca, pz):
    k = 0
    topics = []
    for l in open(pt):
        vs = [float(v) for v in l.split()]
        wvs = zip(range(len(vs)), vs)   # (for a specific topic, wvs store (wordid, propability) )
        wvs = sorted(wvs, key=lambda d:d[1], reverse=True)

        # tmps = ' '.join(['%s:%f' % (voca[w],v) for w,v in wvs[:10]])
        tmps = [(voca[w],v) for w,v in wvs[:15]]
        topics.append((k, tmps))
        k += 1
    return topics

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(10, 40)


def wordcloud(topics,k):
    from wordcloud.wordcloud import WordCloud

    for label, freqs in topics:
        highlight_words = [];
        wordcloud = WordCloud(color_func = grey_color_func, random_state=1, margin=10, background_color='white').fit_words(freqs)

        wordcloud.to_file("./intermediate_data/figures/BTM_wordcould/" + str(k) + "tp/hpv.%s.tagcloud.png"%(label))

def generate_corpus_for_quality_evaluation(k,pz_d,tweets):
    all_tweets = []
    with open(tweets) as f:
        for line in f:
            all_tweets.append(line.strip())

    tweets_pz_d = []
    with open(pz_d) as f:
        for l in f:
            line = l.strip().split(' ')
            tweets_pz_d.append([float(p) for p in line])

    results = {}
    for j in range(len(tweets_pz_d)):
        if 'nan' not in tweets_pz_d[j] and '-nan' not in tweets_pz_d[j]:
            sorted_pz_ds = list(tweets_pz_d[j])
            sorted_pz_ds.sort(reverse = True)
            topic_id = tweets_pz_d[j].index(sorted_pz_ds[0])
            if topic_id not in results:
                # results[topic_id] = [{sorted_pz_ds[0] : all_tweets[j]}]
                results[topic_id] = [all_tweets[j]]
            else:
                # results[topic_id].append({sorted_pz_ds[0] : all_tweets[j]})
                results[topic_id].append(all_tweets[j])

    final_result = {}
    for tp in results:
        temp = []
        samples_number = random.sample(range(1, len(results[tp])), 10)
        for i in samples_number:
            temp.append(results[tp][i])
        final_result[tp] = temp
    to_csv(final_result, './intermediate_data/analysis/BTM/quality_evaluation/'+str(k) + 'tp.csv')

fieldnames = ['topic_id', 'clean_text']
def to_csv(results, csv_output_file):
        with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            # for tp in results:
            #     for tweets in results[tp]:
            #         writer.writerow({
            #             'topic_id': tp,
            #             'clean_text': tweets})
            for tweet in results:
                    writer.writerow({
                        'topic_id': tweet['topic_id'],
                        'clean_text': tweet['clean_text']})


def transfer_to_word_id(input_f, output_f, k):
    voca = {}
    with open('./Biterm/output/' + str(k) + 'tp/voca.txt', 'r') as vc:
        for l in vc:
            wid, w = l.strip().split('\t')[:2]
            voca[w] = int(wid)

    tweets = []
    with open(input_f, 'r',encoding='utf-8') as clusters:
        for line in clusters:
            ws = line.strip().split()
            tweets.append(ws)
    with open(output_f,'w') as w_id:
        for tweet in tweets:
            for w in tweet:
                if w in voca:
                    w_id.write(str(voca[w]) + ' ')
            w_id.write('\n')

def cutoffline_stats(input_f, n):
    pz_d = []
    with open(input_f, 'r') as f:
        for line in f:
            pz_d.append(line.strip().split(' '))

    cnt = 0
    for p_topics in pz_d:
        for p_topic in p_topics:
            if float(p_topic) >= n:
                cnt += 1
                break

    logger.info(cnt)

def generate_cutoffline_file(pz_d,csv_f,output_f, cutoffline):
    result = []
    tweet_topics_distribution = []
    with open(pz_d, 'r') as pz_d_f:
        for l in pz_d_f:
            topic_id = []
            line = l.strip().split(' ')
            for i in range(len(line)):
                if float(line[i]) >= cutoffline:
                    topic_id.append(i)
            tweet_topics_distribution.append(topic_id)

    df = pd.read_csv(csv_f,encoding='utf-8')
    for index, row in df.iterrows():
        temp ={}
        temp['clean_text'] = row['clean_text']
        temp['topic_id'] = tweet_topics_distribution[index]
        result.append(temp)
    to_csv(result, output_f+str(cutoffline)+'.csv')

if __name__ == "__main__":

    logger.info(sys.version)

    # for K in range(6,7):

    #     model_dir = 'Biterm/output/%dtp/model/' % K
    #     voca_pt = 'Biterm/output/%dtp/voca.txt' % K
    #     voca = read_voca(voca_pt)

    #     pz_pt = model_dir + 'k%d.pz' % K
    #     pz = read_pz(pz_pt)

    #     zw_pt = model_dir + 'k%d.pw_z' %  K
    #     topics = dispTopics(zw_pt, voca, pz)

    #     tweets = 'Biterm/sample-data/hpv_tweets.txt'
    #     pz_d = model_dir + 'k%d.pz_d' % K
        # get wordcould figures
        # wordcloud(topics, K)

        # generate corpus for evaluating quality of K
        # generate_corpus_for_quality_evaluation(K,pz_d,tweets)

    #transfer tweets txt to word id file
    # transfer_to_word_id('./intermediate_data/analysis/BTM/cutoffline_annotation/random_100.txt', './intermediate_data/analysis/BTM/cutoffline_annotation/100_doc_wids.txt', 7)
    # transfer_to_word_id('./intermediate_data/preprocessed_text_and_geo.txt','./intermediate_data/word_id.txt',7)


    # test different cutoffline to count tweets
    # n = sys.argv[1]
    # cutoffline_stats('./intermediate_data/k7.pz_d', float(n))

    # generate annotation csv for different cutoffline
    cutoffline = float(sys.argv[1])
    generate_cutoffline_file('intermediate_data/analysis/BTM/cutoffline_annotation/100_k7.pz_d','intermediate_data/analysis/BTM/cutoffline_annotation/random_100.csv','intermediate_data/analysis/BTM/cutoffline_annotation/annotation/', cutoffline)