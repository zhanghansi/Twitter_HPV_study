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
import Arun2010
import CaoJuan2009
import Deveaud2014
import csv

import operator

import pprint
pp = pprint.PrettyPrinter(indent=4)


import nltk
from nltk.stem import WordNetLemmatizer


import gensim
from gensim import corpora, models, similarities

import numpy as np
import matplotlib.pyplot as plt

import random

def load_stoplist():
    stoplist = set()
    with open('./english.stop.txt', newline='', encoding='utf-8') as f:
        for line in f:
            stoplist.add(line.strip())
    return stoplist

def filter_by_frequency(texts, cnt = 3):
    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    return [[token for token in text if frequency[token] > 1] for text in texts]

# pattern = r'''(?x)    # set flag to allow verbose regexps
#     [A-Z]\.+\S+        # abbreviations, e.g. U.S.A.
#     | \w+-\w+\S+        # words with optional internal hyphens
# '''
    # | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    # | \.\.\.            # ellipsis
    # | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
# >>> nltk.regexp_tokenize(text, pattern)

pattern=r'[A-Z]\.+\S+|\w+\-\w+|\w+'

def load_data(csv_file, text_fields = []):

    data = pd.read_csv(csv_file, encoding='utf-8')
    
    documents = []
    for i, r in data.iterrows():

        document = ''
        for text_field in text_fields:
            #logger.info(pd.isnull(r[text_field]))
            if(pd.notnull(r[text_field])):
                # document = '%s  %s'%(document, common.cleanhtml(common.remove_hashtag_sign(common.remove_username(common.remove_url(ftfy.fix_text(r[text_field]))))))
                document = '%s  %s'%(document, r[text_field])

        documents.append(document)

    logger.info("# of documents: %d"%len(documents))

    stoplist = load_stoplist()
    # logging.info(stoplist)
    # quit()
    wordnet_lemmatizer = WordNetLemmatizer()

    texts = [[wordnet_lemmatizer.lemmatize(word.lower()) for word in nltk.regexp_tokenize(document, pattern) if wordnet_lemmatizer.lemmatize(word.lower()) not in stoplist] for document in documents]

    # for text in texts:
    #     while ("vaccine" in text):
    #         text.remove('vaccine')
    #     while ("cancer" in text):
    #         text.remove('vaccine')
        # logger.info(text)

    # # quit()

    texts = filter_by_frequency(texts)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # logger.info(corpus[0])
    return dictionary, corpus


def train_num_of_topics(dictionary, corpus, ranges, output_figure):

    #random.shuffle(corpus)

    test_l = int(len(corpus) * 0.2)
    train_corpus = corpus[test_l+1:]
    test_corpus = corpus[:test_l]

    stats = {
        'perplex': [],
        'per_word_perplex': [],
        'log_perplexity': []
    }
    num_topics_list = ranges #range(5, 151, 5)

    number_of_words = sum(cnt for document in test_corpus for _, cnt in document)
    for num_topics in num_topics_list:

        lda = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=dictionary, num_topics=num_topics, eval_every=10, alpha='auto', chunksize=10000, passes=10)

        perplex = lda.bound(test_corpus)
        #logger.info(perplex)
        
        per_word_perplex = np.exp2(-perplex / number_of_words) 
        log_perplexity = lda.log_perplexity(test_corpus)

        stats['perplex'].append(perplex)
        stats['per_word_perplex'].append(per_word_perplex)
        stats['log_perplexity'].append(log_perplexity)

    #logger.info(stats)
    # for num_topics in num_topics_list:
    #     logger.info('[%d]: perplex: %.3f; per_world_perlex: %.3f'%(num_topics, stats['perplex'][num_topics], stats['per_word_perplex']))

    ax = plt.figure(figsize=(7, 4), dpi=600).add_subplot(111)
    plt.plot(num_topics_list, stats['per_word_perplex'], color="#254F09")
    plt.xlim(num_topics_list[0], num_topics_list[-1])
    plt.ylabel('Per Word Perplexity')
    plt.xlabel('topics')
    plt.title('')
    plt.savefig(output_figure, format='png', bbox_inches='tight', pad_inches=0.1)
    #plt.show()

def lda_train(dictionary, corpus, num_topics = 15):
    number_of_words = sum(cnt for document in corpus for _, cnt in document)

    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, eval_every=10, alpha='auto', chunksize=10000, passes=10)

    lda.print_topics(num_topics=num_topics)
    return lda

def lda_infer(lda_model_file, csv_file, dictionary):
    pass
    

def highlight_color_func(word, highlight_words, font_path, font_size, position, orientation, random_state=None):
    if random_state is None:
        random_state = random.Random()

    if (word in highlight_words):
        return "hsl(0, 100%, 50%)"
    else:
        return "hsl(0, 0%%, %d%%)" % random.randint(10, 40)


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(10, 40)

# def wordcloud(topics, output_figure):
#     from wordcloud import WordCloud
#     import matplotlib.pyplot as plt

#     for label, freqs in topics:
#         wordcloud = WordCloud(margin=5,random_state=1,background_color='white').fit_words(freqs)
#         logger.info(wordcloud)
#         plt.figure()
#         plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3))
#         plt.axis("off")
#         plt.savefig("%s.%s.tagcloud.png"%(output_figure, label))
#         plt.close('all')

def wordcloud(topics=[]):
    from wordcloud.wordcloud import WordCloud

    for label, freqs in topics:
        # logger.info(label)
        # logger.info(freqs[0])
        # quit()
        highlight_words = [];
        wordcloud = WordCloud(color_func = grey_color_func, random_state=1, margin=10, background_color='white').fit_words(freqs)
        # wordcloud.to_file("./all_data/figures/adv_in_nj/hpv.%s.tagcloud.png"%(label))
        wordcloud.to_file("./intermediate_data/laypeople/30tp/hpv.%s.tagcloud.png"%(label))

def fix_utf8(csv_file):

    with open(csv_file, 'rb') as rf, open('%s.fixed.utf-8'%csv_file, 'w', encoding='utf-8') as wf:
        f = rf.read().decode('utf-8', 'ignore')
        wf.write(ftfy.fix_text(f))

fieldnames = ['id', 'text', 'clean_text', 'place', 'user_location', 'us_state', 'created_at', 'username', 'user_id','classifier','topic']
def to_csv(tweets, csv_output_file):
        with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for tweet in tweets:

                writer.writerow(tweet)

def topic_distribution(lda_model, dictionary, path):
    tweets = []
    with open(path, 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        # prob_result = []
        cnt = 0
        for row in reader:
            stoplist = load_stoplist()
            wordnet_lemmatizer = WordNetLemmatizer()
            text = [wordnet_lemmatizer.lemmatize(word.lower()) for word in nltk.regexp_tokenize(row['clean_text'], pattern) if wordnet_lemmatizer.lemmatize(word.lower()) not in stoplist]
            doc_bow = dictionary.doc2bow(text)
            doc_lda = lda_model[doc_bow]
            row['topic'] = []
            # logger.info(doc_lda)
            for tp in doc_lda:
                if tp[1] >= 0.25:
                # if tp[1] >= 0.35 and tp[1] < 0.4:
                    # logger.info(row['text'])
                    # logger.info(tp[0])
                    row['topic'].append(tp[0])
            # if len(row['topic']) >= 2:
            #     cnt += 1
            if len(row['topic']) != 0 :
                cnt += 1
            tweets.append(row)
            # if cnt == 5000:
            #     break
            # logger.info(topic_id)
            # prob_result.append(max_prob)
        to_csv(tweets, './intermediate_data/promotional/cutoffline/promotional_topics_0.25.csv')
        logger.info(cnt)


def plot_topics(x,c_y,d_y,a_y):
    plt.plot(x,c_y)
    plt.plot(x,d_y)
    plt.plot(x,a_y)
    plt.xlabel('#topics')
    plt.legend(['CaoJuan2009', 'Deveaud2014', 'Arun2010'], loc='upper right')
    plt.savefig('./intermediate_data/promotional/#_of_topics', format='png', bbox_inches='tight', pad_inches=0.1)
    # plt.show()

def find_topics(dictionary, corpus, topics):
    a_y = []
    d_y = []
    c_y = []
    for num_topic in topics:
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topic, eval_every=10, alpha='auto', chunksize=10000, passes=10)
        a_y.append(Arun2010.FindTopicsNumber(dictionary, corpus, num_topic, model))
        d_y.append(Deveaud2014.FindTopicsNumber(dictionary, corpus, num_topic, model))
        c_y.append(CaoJuan2009.FindTopicsNumber(dictionary, corpus, num_topic, model))
    c_y = [(y / max(c_y) * 10) for y in c_y]
    d_y = [(y / max(d_y) * 10) for y in d_y]
    c_y = [(y / max(d_y) * 10) for y in c_y]
    logger.info(a_y)
    logger.info(d_y)
    logger.info(c_y)
    plot_topics(topics,c_y,d_y,a_y)

topic_stats_filenames = ['topic_id','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th']
def to_csv_topic_stats(topics, path):
    with open(path, 'w', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=topic_stats_filenames, delimiter=',', quoting=csv.QUOTE_ALL)

            writer.writeheader()
            for topic_tuple in topics:
                writer.writerow({'topic_id': topic_tuple[0],
                                '1st': topic_tuple[1][0][0] + '(' + str(round(topic_tuple[1][0][1],4)) + ')',
                                '2nd': topic_tuple[1][1][0] + '(' + str(round(topic_tuple[1][1][1],4)) + ')',
                                '3rd': topic_tuple[1][2][0] + '(' + str(round(topic_tuple[1][2][1],4)) + ')',
                                '4th': topic_tuple[1][3][0] + '(' + str(round(topic_tuple[1][3][1],4)) + ')',
                                '5th': topic_tuple[1][4][0] + '(' + str(round(topic_tuple[1][4][1],4)) + ')',
                                '6th': topic_tuple[1][5][0] + '(' + str(round(topic_tuple[1][5][1],4)) + ')',
                                '7th': topic_tuple[1][6][0] + '(' + str(round(topic_tuple[1][6][1],4)) + ')',
                                '8th': topic_tuple[1][7][0] + '(' + str(round(topic_tuple[1][7][1],4)) + ')',
                                '9th': topic_tuple[1][8][0] + '(' + str(round(topic_tuple[1][8][1],4)) + ')',
                                '10th': topic_tuple[1][9][0] + '(' + str(round(topic_tuple[1][9][1],4)) + ')',
                                })


if __name__ == "__main__":

    logger.info(sys.version)


    # twitter data
    # BASE = './all_data/Data_state/NJ/NJ_ad'
    BASE = './intermediate_data/laypeople/laypeople'
    # BASE = './intermediate_data/promotional/promotional'
    INPUT = '%s.csv'%BASE
    TEXT_FIELDS = ['clean_text']
    SOURCE = 'twitter'


    DICT = '%s.hpv.dict'%BASE
    MM = '%s.hpv.mm'%BASE

    # fix_utf8(INPUT)
    # quit()
    # Step 1: Create dictionary and corpus
    # dictionary, corpus = load_data(INPUT, TEXT_FIELDS)
    # dictionary.save(DICT)
    # corpora.MmCorpus.serialize(MM, corpus)
    # quit()


    # Step 2:
    # dictionary = corpora.Dictionary.load(DICT)
    # corpus = corpora.MmCorpus(MM)
    # find_topics(dictionary, corpus, range(20, 80, 3))
    # train_num_of_topics(dictionary, corpus, range(10, 40, 1), '%s.train_num_of_topics_by_perplexity.png'%INPUT)
    # quit()

    # Step 3: train lda
    dictionary = corpora.Dictionary.load(DICT)
    corpus = corpora.MmCorpus(MM)

    num_topics = 30
    lda = lda_train(dictionary, corpus, num_topics = num_topics)
    MODEL = '%s.%d.lda.model'%(BASE,num_topics)
    lda.save(MODEL)

    # lda = gensim.models.ldamodel.LdaModel.load(MODEL)

    # wordcloud(lda.show_topics(num_topics=num_topics, formatted=False), './figures/%s'%(SOURCE))
    # wordcloud(topics=lda.show_topics(num_topics=num_topics, formatted=False))
    # quit()
    # topics=lda.show_topics(num_topics=num_topics, num_words=10, formatted=False)
    # logger.info(topics)
    # to_csv_topic_stats(topics,'./intermediate_data/analysis/laypeople_topics_keywords_stats.csv')

    # step 4: lda_distribution
    # topic_distribution(lda,dictionary,'./intermediate_data/promotional/promotional.csv')
