#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


import os
import re
import json
import sys
import time
import math
import csv
import common

import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stats
import numpy as np

# from lineplot import lineplot

google_parse_time_format = '%m/%d/%Y'
states = ["ny", "mo", "pa", "nj", "ri", "wa", "ca", "il", "ky", "tx", "fl", "dc", "co", "ct", "va", "oh", "in", "ma", "mi", "ok", "me", "ms", "tn", "ga", "nc", "ia", "ut", "mn", "md", "wi", "ne", "sc", "sd", "or", "ks", "mt", "az", "nv", "nh", "nd", "hi", "al", "ar", "vt", "nm", "la", "ak", "de", "id", "wv", "wy"]
def count_tweets_by_month_state(csv_file,states):
    months = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_f:
            reader = csv.DictReader(csv_f)
            count_result = {}
            for tweet in reader:
                month = common.time_str_to_month(tweet['created_at'])
                if(month not in months):
                    months.append(month)
                else:
                    continue
    count_result = {};
    for state in states:
        count_result[state] = []
        counter = {}
        for month in months:
            counter[month] = 0;
        count_result[state] = counter
    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_f:
            reader = csv.DictReader(csv_f)
            for tweet in reader:
                count_result[tweet['us_state']][common.time_str_to_month(tweet['created_at'])] += 1
    #logging.info(count_result)
    return count_result
def count_tweets_by_month(csv_file):
    count_result = {}
    # with open(csv_file, 'r', newline='', encoding='utf-8') as csv_f:
    #           reader = csv.DictReader(csv_f)
    #           id = []
    #           for tweet in reader:
    #             if tweet['id'] not in id:
    #                 id.append(tweet['id'])
    #             else:
    #                 logger.info(tweet['user_id'])
    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_f:
            reader = csv.DictReader(csv_f)
            count_result = {}
            for tweet in reader:
                month = common.time_str_to_month(tweet['created_at'])
                if(month in count_result):
                    count_result[month] += 1;
                else:
                    count_result[month] = 1;
    #logging.info(count_result)
    return count_result

def count_tweets_by_state(csv_file):
    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_f:
            reader = csv.DictReader(csv_f)
            count_result = {}
            for tweet in reader:
                state = tweet['us_state']
                if (state in count_result):
                    count_result[state] += 1
                else:
                    count_result[state] = 1
            return count_result

def twitter_stats_by_week(csv_file, stats_folder, weekdays = {}):
    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        col = {}
        for tweet in reader:
            #week = common.time_str_to_week(tweet['created_at'])
            day = common.time_str_to_day(tweet['created_at'])
            created_at = tweet['created_at']
            
            if (day not in col):
                col[day] = {}

            if (created_at not in col[day]):
                col[day][created_at] = 0

            col[day][created_at] += 1

    stats = []
    for root, dirs, files in os.walk(os.path.abspath(stats_folder)):
        for f in files:
            if (f.endswith('.json')):
                with open(os.path.join(root, f), 'r') as infile:
                    stats.append(json.load(infile))

    stats_by_week = {}

    for week_start in weekdays:

        counts = {
                'dates': weekdays[week_start],
                'mention_sum': 0,
                'total_sum': 0
            }
        # if (week_start not in stats_by_week):
        #     stats_by_week[week_start] = {
        #         'dates': weekdays[week_start],
        #         'mention_sum': 0,
        #         'total_sum': 0
        #     }

        for day in weekdays[week_start]:
            if (day in col):
                for created_at in col[day]:
                    counts['mention_sum'] += col[day][created_at]

            for stat in stats:
                if (day in stat):
                    try:
                        counts['total_sum'] += stat[day]['total']['after_filter']
                    except Exception as exc:
                        logger.info(stat[day])
                        logger.warn('ignore: %s'%(exc))
                        quit()

        if (counts['mention_sum'] > 0 and counts['total_sum'] > 0):
            stats_by_week[week_start] = counts

    return stats_by_week

def google_trends_stats_by_week(csv_file):
    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        data = {}
        for row in reader:

            week_start = row['Week']

            data[week_start] = row['hpv']

        return data


def get_weekdays(week_strarts):

    weekdays = {}
    for week_start in week_strarts:

        weekdays[week_start] = common.time_str_to_weekday(week_start, parse_time_format=google_parse_time_format)

    return weekdays

def plot_by_month():
    google_data = google_trends_stats_by_week('./data/google.trends.iot.2009-2015.csv')

    weekdays = get_weekdays(google_data.keys())

    twitter_data = twitter_stats_by_week('./data/iot.csv', './data/stats/', weekdays=weekdays)

    keys = sorted(twitter_data.keys())

    del keys[-1]

    monthly_data = {}
    for key in keys:

        cur_year_month = '%s-%s'%(key.split('-')[0], key.split('-')[1])

        if (cur_year_month not in monthly_data):
            monthly_data[cur_year_month] = {
                'mention_sum': 0,
                'total_sum': 0
            }

        monthly_data[cur_year_month]['mention_sum'] += twitter_data[key]['mention_sum']
        monthly_data[cur_year_month]['total_sum'] += twitter_data[key]['total_sum']

    year_months = sorted(monthly_data.keys())

    logger.info(year_months)

    quit()
    data = []
    for month in year_months:
        data.append(float(monthly_data[month]['mention_sum'])/monthly_data[month]['total_sum'])

    min_d = min(data)
    max_d = max(data)
    
    scaled_data = []
    for d in data:
        scaled_data.append((d - min_d)/(max_d - min_d)*100)

    monthly_google_data = {}
    for key in keys:
        cur_year_month = '%s-%s'%(key.split('-')[0], key.split('-')[1])

        if (cur_year_month not in monthly_google_data):
            monthly_google_data[cur_year_month] = {
                'sum': 0,
                'count': 0
            }

        monthly_google_data[cur_year_month]['sum'] += int(google_data[key])
        monthly_google_data[cur_year_month]['count'] += 1

    scaled_google_data = []
    for month in year_months:
        scaled_google_data.append(float(monthly_google_data[month]['sum'])/monthly_google_data[month]['count'])

    # tick
    xticklabels = []
    full_xticklabels = []
    looking_for = ['2015', '2014', '2010', '2009']

    for month in year_months:

        full_xticklabels.append(month)
        found = False
        for s in looking_for:
            if (month.startswith(s)):
                xticklabels.append(s)
                looking_for.pop()
                found = True
                break

        if (not found):
            xticklabels.append('')
    classes = {
        'twitter': {
            'label': 'Twitter',
            'color': 'g'
        },
        'google':{
            'label': 'Google Trends',
            'color': 'b'
        }
    }

    full_data = {
        'twitter': {},
        'google': {}
    }

    for key, twitter_value, google_value in zip(year_months, scaled_data, scaled_google_data):
        full_data['twitter'][key] = math.ceil(twitter_value)
        full_data['google'][key] = math.ceil(google_value)
        

    plt = lineplot(full_data, classes=classes, xticklabels = xticklabels, full_xticklabels = full_xticklabels)

    logger.info(stats.pearsonr(scaled_data, scaled_google_data))

    plt.savefig('./figures/iot.twitter.trend.monthly.png')


def plot_by_month_v2():
    google_data = google_trends_stats_by_week('./data/google.trends.hpv.2015-2016.csv')

    weekdays = get_weekdays(google_data.keys())

    twitter_data = twitter_stats_by_week('./data/hpv/hpv.csv', './data/hpv/', weekdays=weekdays)

    keys = sorted(twitter_data.keys())

    del keys[-1]

    monthly_data = {}
    for key in keys:

        cur_year_month = '%s-%s'%(key.split('/')[2], key.split('/')[0])

        if (cur_year_month not in monthly_data):
            monthly_data[cur_year_month] = {
                'mention_sum': 0,
                'total_sum': 0
            }

        monthly_data[cur_year_month]['mention_sum'] += twitter_data[key]['mention_sum']
        monthly_data[cur_year_month]['total_sum'] += twitter_data[key]['total_sum']

    year_months = list(monthly_data.keys())

    missing_year_months = ['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08',
        '2015-09', '2015-10','2016-12']
    year_months.extend(missing_year_months)
    year_months = sorted(year_months)
    
    data = []
    for month in year_months:
        if (month in missing_year_months):
            data.append(np.nan)
        else:
            data.append(float(monthly_data[month]['mention_sum'])/monthly_data[month]['total_sum'])

    min_d = np.nanmin(data)
    max_d = np.nanmax(data)

    scaled_data = []
    for d in data:
        if (math.isnan(d)):
            scaled_data.append(np.nan)
        else:
            scaled_data.append((d - min_d)/(max_d - min_d)*100)

    monthly_google_data = {}
    for key in keys:
        cur_year_month = '%s-%s'%(key.split('/')[2], key.split('/')[0])

        if (cur_year_month not in monthly_google_data):
            monthly_google_data[cur_year_month] = {
                'sum': 0,
                'count': 0
            }

        monthly_google_data[cur_year_month]['sum'] += int(google_data[key])
        monthly_google_data[cur_year_month]['count'] += 1

    scaled_google_data = []
    for month in year_months:
        if (month in missing_year_months):
            scaled_google_data.append(np.nan)
        else:
            scaled_google_data.append(float(monthly_google_data[month]['sum'])/monthly_google_data[month]['count'])

    # tick
    xticklabels = []
    full_xticklabels = []
    looking_for = ['2015', '2016']

    for month in year_months:

        full_xticklabels.append(month)
        found = False
        for s in looking_for:
            if (month.startswith(s)):
                xticklabels.append(s)
                looking_for.pop()
                found = True
                break

        if (not found):
            xticklabels.append('')
    classes = {
        'twitter': {
            'label': 'Twitter',
            'color': 'g'
        },
        'google':{
            'label': 'Google Trends',
            'color': 'b'
        }
    }

    full_data = {
        'twitter': {},
        'google': {}
    }

    for key, twitter_value, google_value in zip(year_months, scaled_data, scaled_google_data):
        if (math.isnan(twitter_value)):
            full_data['twitter'][key] = np.nan
        else:
            full_data['twitter'][key] = math.ceil(twitter_value)

        if (math.isnan(google_value)):
            full_data['google'][key] = np.nan
        else:
            full_data['google'][key] = math.ceil(google_value)

    plt = lineplot(full_data, classes=classes, xticklabels = xticklabels, full_xticklabels = full_xticklabels, xxlabel='Year', yylabel='RMV (Twitter) / RSV (Google)')

    logger.info(stats.pearsonr(scaled_data, scaled_google_data))

    plt.savefig('./figures/hpv.twitter.trend.monthly.v2.png')

def plot_by_month_volume():


    google_data = google_trends_stats_by_week('./data/google.trends.iot.2009-2015.csv')

    weekdays = get_weekdays(google_data.keys())

    twitter_data = twitter_stats_by_week('./data/iot.csv', './data/stats/', weekdays=weekdays)

    keys = sorted(twitter_data.keys())

    del keys[-1]

    monthly_data = {}
    for key in keys:

        cur_year_month = '%s-%s'%(key.split('-')[0], key.split('-')[1])

        if (cur_year_month not in monthly_data):
            monthly_data[cur_year_month] = {
                'mention_sum': 0,
                'total_sum': 0
            }

        monthly_data[cur_year_month]['mention_sum'] += twitter_data[key]['mention_sum']
        monthly_data[cur_year_month]['total_sum'] += twitter_data[key]['total_sum']

    year_months = list(monthly_data.keys())

    missing_year_months = ['2009-01', '2009-02', '2009-03', '2009-04', '2010-11', '2010-12', '2014-11', '2014-12',
        '2011-01', '2011-02', '2011-03', '2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11', '2011-12',
        '2012-01', '2012-02', '2012-03', '2012-04', '2012-05', '2012-06', '2012-07', '2012-08', '2012-09', '2012-10', '2012-11', '2012-12',
        '2013-01', '2013-02', '2013-03', '2013-04', '2013-05', '2013-06', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12']
    year_months.extend(missing_year_months)
    year_months = sorted(year_months)

    data = []
    for month in year_months:
        if (month in missing_year_months):
            data.append(np.nan)
        else:
            data.append(float(monthly_data[month]['total_sum']))

    # tick
    xticklabels = []
    full_xticklabels = []
    looking_for = ['2015', '2014', '2013', '2012', '2011', '2010', '2009']

    for month in year_months:

        full_xticklabels.append(month)
        found = False
        for s in looking_for:
            if (month.startswith(s)):
                xticklabels.append(s)
                looking_for.pop()
                found = True
                break

        if (not found):
            xticklabels.append('')
    classes = {
        'twitter': {
            'label': 'Twitter',
            'color': 'g'
        }
    }

    full_data = {
        'twitter': {}
    }

    for key, twitter_value in zip(year_months, data):
        if (math.isnan(twitter_value)):
            full_data['twitter'][key] = np.nan
        else:
            full_data['twitter'][key] = math.ceil(twitter_value)

    plt = lineplot(full_data, classes=classes, xticklabels = xticklabels, full_xticklabels = full_xticklabels, xxlabel='Year', yylabel='RMV (Twitter) / RSV (Google)')

    plt.savefig('./figures/iot.twitter.trend.monthly.volume.v2.png')

def plot_by_week():
    google_data = google_trends_stats_by_week('./data/google.trends.iot.2009-2015.csv')

    weekdays = get_weekdays(google_data.keys())

    twitter_data = twitter_stats_by_week('./data/iot.csv', './data/stats/', weekdays=weekdays)

    keys = sorted(twitter_data.keys())

    del keys[-1]

    xticklabels = []
    full_xticklabels = []
    data = []
    looking_for = ['2015', '2014', '2010', '2009']
    for key in keys:
        data.append(float(twitter_data[key]['mention_sum'])/twitter_data[key]['total_sum'])
        full_xticklabels.append(key)
        found = False
        for s in looking_for:
            if (key.startswith(s)):
                xticklabels.append(s)
                looking_for.pop()
                found = True
                break

        if (not found):
            xticklabels.append('')


    min_d = min(data)
    max_d = max(data)
    
    scaled_data = []
    for d in data:
        scaled_data.append((d - min_d)/(max_d - min_d)*100)

    classes = {
        'twitter': {
            'label': 'Twitter',
            'color': 'g'
        },
        'google':{
            'label': 'Google Trends',
            'color': 'b'
        }
    }

    full_data = {
        'twitter': {},
        'google': {}
    }

    for key, value in zip(keys, scaled_data):
        full_data['twitter'][key] = math.ceil(value)

    google_d = []
    for key in keys:
        full_data['google'][key] = int(google_data[key])
        google_d.append(int(google_data[key]))

    #plt = lineplot(full_data, classes=classes, xticklabels = xticklabels, full_xticklabels = full_xticklabels)

    logger.info(stats.pearsonr(scaled_data, google_d))

    #plt.savefig('./figures/iot.twitter.trend.png')

def split():
    google_data = google_trends_stats_by_week('./data/google.trends.iot.2009-2015.csv')

    weekdays = get_weekdays(google_data.keys())

    twitter_data = twitter_stats_by_week('./data/iot.csv', './data/stats/', weekdays=weekdays)

    keys = sorted(twitter_data.keys())

    del keys[-1]

    with open('./data/iot.csv', 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        col = {}
        for tweet in reader:
            #week = common.time_str_to_week(tweet['created_at'])
            day = common.time_str_to_day(tweet['created_at'])

            if (day not in col):
                col[day] = []

            col[day].append(tweet)


    for key in keys:
        for day in weekdays[key]:
            if (day in col):
                for tweet in col[day]:
                    with open('./data/txt/iot_%s.txt'%(key), 'a') as txt_f:
                        txt_f.write('%s  \n'%tweet['text'])

def get_keys():
    google_data = google_trends_stats_by_week('./data/google.trends.iot.2009-2015.csv')

    weekdays = get_weekdays(google_data.keys())

    twitter_data = twitter_stats_by_week('./data/iot.csv', './data/stats/', weekdays=weekdays)

    keys = sorted(twitter_data.keys())

    del keys[-1]

    return keys

def plot_google_data():
    google_data = google_trends_stats_by_week('./data/google.trends.iot.2009-2015.csv')

    keys = sorted(get_weekdays(google_data.keys()))

    xticklabels = []
    full_xticklabels = []
    looking_for = ['2015', '2014', '2013', '2012', '2011', '2010', '2009']
    for key in keys:
        full_xticklabels.append(key)
        found = False
        for s in looking_for:
            if (key.startswith(s)):
                xticklabels.append(s)
                looking_for.pop()
                found = True
                break

        if (not found):
            xticklabels.append('')

    classes = {
        'google':{
            'label': 'Google Trends',
            'color': 'b'
        }
    }

    full_data = {
        'google': {}
    }

    google_d = []
    for key in keys:
        full_data['google'][key] = int(google_data[key])
        google_d.append(int(google_data[key]))

    plt = lineplot(full_data, classes=classes, xxlabel='Years', yylabel='Percent of Tweets/Google Searches', xticklabels = xticklabels, full_xticklabels = full_xticklabels)

    plt.savefig('./figures/iot.google.trend.png')

def count(stats_folder):
    stats = []
    for root, dirs, files in os.walk(os.path.abspath(stats_folder)):
        for f in files:
            if (f.endswith('.json')):
                with open(os.path.join(root, f), 'r') as infile:
                    stats.append(json.load(infile))

    before_filter = 0
    after_filter = 0

    for stat in stats:
        for day in stat:

            before_filter += stat[day]['total']['before_filter']
            after_filter += stat[day]['total']['after_filter']

    logger.info(before_filter)
    logger.info(after_filter)

# fieldnames = ['state', '201511', '201512', '201601', '201602', '201603', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612', '201701', '201702', '201703']

def to_csv_state_month(count_result, csv_output_file):
    #tweets = []
    months = ['201511', '201512', '201601', '201602', '201603', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612', '201701', '201702', '201703']
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
        
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)

        writer.writeheader()
        for state in count_result:
            # logger.info(state['201511'])
            writer.writerow({'state': state,
                             '201511': count_result[state]['201511'],
                             '201512': count_result[state]['201512'],
                             '201601': count_result[state]['201601'],
                             '201602': count_result[state]['201602'],
                             '201603': count_result[state]['201603'],
                             '201604': count_result[state]['201604'],
                             '201605': count_result[state]['201605'],
                             '201606': count_result[state]['201606'],
                             '201607': count_result[state]['201607'],
                             '201608': count_result[state]['201608'],
                             '201609': count_result[state]['201609'],
                             '201610': count_result[state]['201610'],
                             '201611': count_result[state]['201611'],
                             '201612': count_result[state]['201612'],
                             '201701': count_result[state]['201701'],
                             '201702': count_result[state]['201702'],
                             '201703': count_result[state]['201703'],})

fieldnames = ['month', 'num_tweets']
def to_csv_month(count_result, csv_output_file):
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
        
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)

        writer.writeheader()
        for month in count_result:
            writer.writerow({'month': month,
                            'num_tweets': count_result[month]})

fieldnames_state = ['state', 'num_tweets']
def to_csv_state(count_result, csv_output_file):
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
        
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames_state, delimiter=',', quoting=csv.QUOTE_ALL)

        writer.writeheader()
        for state in count_result:
            writer.writerow({'state': state,
                            'num_tweets': count_result[state]})

def plot_hist(dict):

    data = []
    names = []
    top5 = []
    top5_names = []
    bottom5 = []
    bottom5_names = []
    for name in dict:
        names.append(name)
        data.append(dict[name])
    sorted_data = sorted(data)
    logger.info(sorted_data)
    for n in range(5):
        bottom5.append(sorted_data[n])
        bottom5_names.append(names[data.index(sorted_data[n])])
        top5.append(sorted_data[len(sorted_data)-1-n])
        top5_names.append(names[data.index(sorted_data[len(sorted_data)-1-n])])

    x_names = top5_names + bottom5_names
    y_data = top5 + bottom5
    # logger.info(x_names)
    # logger.info(y_data)
    # quit()
    ax = plt.subplot(111)
    ax.set_xlabel("Top 5 and bottom 5 States")
    ax.set_ylabel("P/N")
    ax.set_ylim([0,2.5])
    width = 0.5
    bins = [* map(lambda x: x-width/2,range(1,len(y_data)+1))]
    ax.bar(bins,y_data,width=width)
    x_cor_top5 = [* map(lambda x: x, range(1,len(x_names)+1))]
    ax.set_xticks(x_cor_top5)
    ax.set_xticklabels(x_names,rotation=45)
    # ax.set_xlim([0,20])
    for x,y in zip(x_cor_top5, y_data):
        ax.text(x, y, str(y), horizontalalignment='center')

    # ax = plt.subplot(122)
    # ax.set_xlabel("Bottom 5 States")
    # ax.set_ylim([0,4])
    # bins = [* map(lambda x: x-width/2,range(1,len(bottom5)+1))]
    # ax.bar(bins,bottom5,width=width)
    # x_cor_bottom5 = [* map(lambda x: x, range(1,len(bottom5)+1))]
    # ax.set_xticks(x_cor_bottom5)
    # ax.set_xticklabels(bottom5_names,rotation=45)
    # for x,y in zip(x_cor_bottom5, bottom5):
    #     ax.text(x, y, str(y), horizontalalignment='center')
    plt.show()


def count_tweets_by_state_senti(csv_file):
    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_f:
            reader = csv.DictReader(csv_f)
            count_result = {}
            for tweet in reader:
                state = tweet['us_state']
                if (state in count_result):
                    count_result[state]['num_tweets'] += 1
                else:
                    count_result[state] = {}
                    count_result[state]['num_tweets'] = 1
                    count_result[state]['positive'] = 0
                    count_result[state]['negative'] = 0
                    count_result[state]['neutral'] = 0
                # logger.info(tweet['senti'])
                if tweet['senti'] == '2':
                    count_result[state]['negative'] += 1
                elif tweet['senti'] == '1':
                    count_result[state]['positive'] += 1
                else:
                    count_result[state]['neutral'] += 1
    senti_fileds = ['state','num_tweets','positive','negative','neutral']
    with open('./all_data/stats/stats_by_state_senti.csv', 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
        
        writer = csv.DictWriter(csv_f, fieldnames=senti_fileds, delimiter=',', quoting=csv.QUOTE_ALL)

        writer.writeheader()
        for state in count_result:
            writer.writerow({'state': state,
                            'num_tweets': count_result[state]['num_tweets'],
                            'positive': count_result[state]['positive'],
                            'negative': count_result[state]['negative'],
                            'neutral': count_result[state]['neutral']})
    return count_result

def merge(path1,path2):
    result = []
    with open(path1, 'r', newline='', encoding='utf-8') as csv_f:
            reader = csv.DictReader(csv_f)
            for tweet in reader:
                result.append(tweet)
    with open(path2, 'r', newline='', encoding='utf-8') as csv_f:
            reader = csv.DictReader(csv_f)
            final = []
            for tweet in reader:
                for t in result:
                    if t['id'] == tweet['id']:
                        t['label'] = tweet['label']
                        final.append(t)
                        # logger.info(tweet['label'])
            to_csv(final,'./all_data/senti_final.csv')
    # logger.info(result)

fieldnames_1 = ['id', 'text', 'clean_text', 'place', 'user_location', 'us_state', 'created_at', 'username', 'user_id','classifier','label']
def to_csv(tweets, csv_output_file):
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
        
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames_1, delimiter=',', quoting=csv.QUOTE_ALL)

        writer.writeheader()
        
        for tweet in tweets:
            # logger.info(tweets)
            writer.writerow(tweet)

if __name__ == "__main__":

    logger.info(sys.version)

    # plot_by_month_volume()
    # quit()

    # count_result = count_tweets_by_month_state('./data/count_statistic/hpv(geo).csv',states)
    # to_csv_state_month(count_result,'./data/count_statistic/tweets_per_state_per_month.csv')
    # quit()

    count_result = count_tweets_by_state('./intermediate_data/hpv_geo.csv')
    # count_result = count_tweets_by_state_senti('./all_data/senti_final.csv')
    # logger.info(count_result)
    # temp = {}
    # for result in count_result:
    #     temp[result] = round((count_result[result]['positive'] / count_result[result]['negative']), 3)

    # merge('./all_data/senti_all/senti.csv','./all_data/hpv_lay_person.predicted.csv')
    # plot_hist(temp)
    to_csv_state(count_result,'./intermediate_data/analysis/tweets_per_state.csv')
    # to_csv_month(count_result,'./data/count_statistic/tweets_per_month(place).csv')
    quit()

    # plot_by_month_v2()
    # quit()
    #plot_google_data()
    #split()
    # count('./data/stats/')
