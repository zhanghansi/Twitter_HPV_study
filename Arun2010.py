import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import sys
import gensim
from gensim import corpora, models, similarities
import numpy as np
import matplotlib.pyplot as plt


def FindTopicsNumber(dictionary, corpus, topics, model):
     # check parameters topics should large than 2
    num_topics_list = topics #range(5, 151, 5)

    l = []
    for document_tuples in corpus:
        document_row = []
        for tup in document_tuples:
            document_row.append(tup[1])
        l.append(np.sum(document_row))
    l_array = np.asarray(l)

    # result_divergence = []
    # for num_topics in num_topics_list:
        # model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, eval_every=10, alpha='auto', chunksize=10000, passes=10)

        ## m1
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
    # m1_matrix = np.exp(np.asmatrix(m1_array)) R model@beta return logarithmized parameters of the word distribution for each topic.
    m1_matrix = np.asmatrix(m1_array)
    cm1 = np.linalg.svd(m1_matrix)[1]

    # ## m2
    m2_tuples = []
    m2 = []
    for document in corpus:
        m2_tuples.append(model.get_document_topics(document))
    for tuples in m2_tuples:
        m2_row = [0.0] * topics
        for tup in tuples:
            m2_row[tup[0]] = tup[1]
        m2.append(m2_row)
    m2_array = np.asarray(m2)
    cm2_array = np.dot(l_array,m2_array)
    norm = np.linalg.norm(l_array, ord = 1)
    cm2 = cm2_array / norm
    # symmetric Kullback-Leibler divergence
    kl_l = np.sum(np.multiply(cm1,np.log(cm1 / cm2)))
    kl_r = np.sum(np.multiply(cm2,np.log(cm2 / cm1)))
    divergence = kl_l + kl_r
        # result_divergence.append(divergence)
        # print(num_topics)
        # print(divergence)
    return divergence
    # plt.plot(num_topics_list,result_divergence)
    # plt.ylabel('divergence')
    # plt.xlabel('#topics')
    # plt.title('')
    # plt.savefig(output_figure, format='png', bbox_inches='tight', pad_inches=0.1)
    # print(result_divergence)