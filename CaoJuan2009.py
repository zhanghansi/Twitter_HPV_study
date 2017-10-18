import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import sys
import gensim
from gensim import corpora, models, similarities
import numpy as np
from itertools import combinations
import scipy.spatial as sp
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
# The pattern is really simple : apply(variable, margin, function).
# –variable is the variable you want to apply the function to.
# –margin specifies if you want to apply by row (margin = 1), by column (margin = 2), or for each element (margin = 1:2). Margin can be even greater than 2, if we work with variables of dimension greater than two.
# –function is the function you want to apply to the elements of your variable.


def FindTopicsNumber(dictionary, corpus, topics, model):
     # check parameters topics should large than 2
    # num_topics_list = topics #range(5, 151, 5)
    # resutls_spy = []
    # resutls_skt = []
    # resutls = []
    # for num_topics in num_topics_list:
        # model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, eval_every=10, alpha='auto', chunksize=10000, passes=10)

    # m1  topic-word matrix
    m1_tuples = []
    m1 = []
    for cnt in range(topics):
        m1_tuples.append(model.get_topic_terms(cnt,topn=90))
    for tuples in m1_tuples:
        m1_row = []
        for tup in tuples:
            m1_row.append(tup[1])
        m1.append(m1_row)
    m1_array = np.asarray(m1)
    # m1_matrix = np.exp(np.asmatrix(m1_array))
    m1_matrix = np.asmatrix(m1_array)
# pair-wise cosine distance
    nrow = m1_matrix.shape[0]
    pairs = list(combinations(range(1,nrow + 1),2))
    pair_1 = []
    pair_2 = []
    for pair in pairs:
        pair_1.append(pair[0]-1)
        pair_2.append(pair[1]-1)
    x = m1_matrix[pair_1, ]
    y = m1_matrix[pair_2, ]
    a = np.dot(x.transpose(),x)
    b = np.dot(y.transpose(),y)
    dist = np.dot(x.transpose(),y) / np.sqrt(np.multiply(a,b))
    # dist_spy = sp.distance.cdist(x, y, 'cosine')
    # dist_skt = sklearn.metrics.pairwise.cosine_distances(x,y)
    # # metric_spy = np.sum(dist_spy) / (num_topics*(num_topics-1) / 2)
    # metric_skt =  np.sum(dist_skt) / (num_topics*(num_topics-1) / 2)
    # # resutls_spy.append(metric_spy)
    # resutls_skt.append(metric_skt)
    # print(metric_spy)
    metric = np.sum(dist) / (topics*(topics-1) / 2)
    # resutls.append(metric)
    # print(metric)
    # print(resutls_spy)
    return metric
    # plt.plot(num_topics_list,resutls)
    # plt.ylabel('cosine distance')
    # plt.xlabel('#topics')
    # plt.title('')
    # plt.savefig(output_figure, format='png', bbox_inches='tight', pad_inches=0.1)
    # print(resutls)