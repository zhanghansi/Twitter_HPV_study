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
        tmps = [(voca[w],v) for w,v in wvs[:10]]
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
        sorted_pz_ds = list(tweets_pz_d[j])
        sorted_pz_ds.sort()
        print(sorted_pz_ds)
        print(tweets_pz_d[j])
        quit()
        for i in range(1):
            topic_id = tweets_pz_d[j].index(sorted_pz_ds[i])
            if topic_id not in results:
                results[topic_id] = [all_tweets[j]]
            else:
                results[topic_id].append([all_tweets[j]])
    print(len(results))

if __name__ == "__main__":

    logger.info(sys.version)

    K = 7

    model_dir = 'Biterm/output/%dtp/model/' % K
    voca_pt = 'Biterm/output/%dtp/voca.txt' % K
    voca = read_voca(voca_pt)

    pz_pt = model_dir + 'k%d.pz' % K
    pz = read_pz(pz_pt)

    zw_pt = model_dir + 'k%d.pw_z' %  K
    topics = dispTopics(zw_pt, voca, pz)

    tweets = 'Biterm/sample-data/hpv_tweets.txt'
    pz_d = model_dir + 'k%d.pz_d' % K
    # get wordcould figures
    # wordcloud(topics, K)

    # generate corpus for evaluating quality of K
    generate_corpus_for_quality_evaluation(K,pz_d,tweets)
