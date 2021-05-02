#!/usr/bin/env python
# coding: utf-8

import numpy as np
np.random.seed(42)
import random
random.seed(42)

from sklearn.datasets import fetch_20newsgroups

import nltk, string, itertools, time
import matplotlib.pyplot as plt

from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

from scipy.optimize import linear_sum_assignment
from umap import UMAP


# Custom stop words
stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(stop_words_skt))


def plot_contingency_table_20(cm, title='Contingency Table', cmap=plt.cm.YlOrBr):
    plt.gcf().clear()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    tick_marks = np.arange(20)
    number_labels = [x for x in range(1,21)]
    plt.xticks(tick_marks, number_labels)
    plt.yticks(tick_marks, number_labels)
    
    plt.ylabel('Cluster Class', fontsize=12)
    plt.xlabel('Actual Class', fontsize=12)
    plt.gcf().set_size_inches(10.0,10.0)
    plt.show()


def k_means_clustering(training_data,
                       target_labels,
                       title='Contingency Matrix',
                       n_clusters=20,
                       random_state=0,
                       max_iter=1000,
                       n_init=30):
    start = time.time()
    km = KMeans(n_clusters=n_clusters,random_state=random_state,max_iter=max_iter,n_init=n_init)
    km.fit(training_data)
    print("Finished clustering in %f seconds" % (time.time()-start))

    cm = contingency_matrix(target_labels, km.labels_)
    # reorder to maximize along diagonal
    rows, cols = linear_sum_assignment(cm, maximize=True)
    new_cm = cm[rows[:,np.newaxis], cols]
    
    print("Show Contingency Matrix:")
    plot_contingency_table_20(new_cm, title=title)
    
    print("Report 5 Measures for K-Means Clustering")
    
    homogeneity = homogeneity_score(target_labels, km.labels_)
    completeness = completeness_score(target_labels, km.labels_)
    v_measure = v_measure_score(target_labels, km.labels_)
    adjusted_rand_index = adjusted_rand_score(target_labels, km.labels_)
    adjusted_mutual_info = adjusted_mutual_info_score(target_labels, km.labels_)

    print("Homogeneity Score: %f" % homogeneity)
    print("Completeness Score: %f" % completeness)
    print("V-Measure Score: %f" % v_measure)
    print("Adjusted Rand Index: %f" % adjusted_rand_index)
    print("Adjusted Mutual Information: %f" % adjusted_mutual_info)
    
    results = {
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "adjusted_rand_index": adjusted_rand_index,
        "adjusted_mutual_info": adjusted_mutual_info }
    
    return results, km


def plot_histogram(title_name, ydata, x_labels = ['1', '2', '3', '5', '10', '20', '50', '100', '300'], 
                   height=range(1,10),xtickangle=0):
    plt.gcf().clear()
    fig, ax = plt.subplots()
    ax.set_xticks([i+0.25 for i in height])
    ax.set_xticklabels(x_labels, fontsize = 12)
    
    rects = plt.bar([i for i in height], ydata, 0.5, align='edge', alpha = 0.8)
    plt.xlabel('Number of Principal Components r', fontsize = 14)
    plt.ylabel('Measure Score', fontsize = 14)
    plt.title(title_name, fontsize = 18)
    plt.axis([0.5,len(x_labels)+1,0,1])
    
    plt.xticks(rotation=xtickangle)
    
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height, '%.3f' % float(height), ha='center', va='bottom')
    
    plt.show()


def svd_dimension_reduction(training_data, n_components=20, random_state=0):
    start = time.time()
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd_dataset = svd.fit_transform(training_data)
    print("NMF complete after %f seconds" % (time.time()-start))
    return svd_dataset, svd


def nmf_dimension_reduction(training_data, n_components=None, solver='cd', beta_loss='frobenius', max_iter=1000, random_state=0):
    start = time.time()
    nmf = NMF(n_components=n_components, solver=solver, init='nndsvda', beta_loss=beta_loss, max_iter=max_iter, random_state=random_state)
    nmf_dataset = nmf.fit_transform(training_data)
    print("NMF complete after %f seconds" % (time.time()-start))
    return nmf_dataset, nmf


def umap_dimension_reduction(training_data, n_components=20, metric='cosine', disconnection_distance=None, random_state=0):
    start = time.time()
    umap = UMAP(n_components=n_components, metric=metric, disconnection_distance=disconnection_distance, random_state=random_state)
    umap_dataset = umap.fit_transform(training_data)
    print("UMAP complete after %f seconds" % (time.time()-start))
    return umap_dataset, umap


# Question 9
dataset_20 = fetch_20newsgroups(subset='all',shuffle=True, remove=('headers', 'footers'), random_state=42)
tfidf_vect_20 = TfidfVectorizer(stop_words=combined_stopwords,min_df=3)
X_train_tfidf_20 = tfidf_vect_20.fit_transform(dataset_20.data) # making the tfidf matrix
print(X_train_tfidf_20.shape)

y_true_20 = dataset_20.target


# Question 9: SVD
svd_homogeneity_1ist = []
svd_completeness_1ist = []
svd_v_measure_1ist = []
svd_adjusted_rand_index_1ist = []
svd_adjusted_mutual_info_1ist = []

for r in [1, 2, 3, 5, 10, 20, 50, 100, 300]:
    print('n_components:',r)
    svd_dataset, svd = svd_dimension_reduction(X_train_tfidf_20, n_components = r, random_state=0)
    k_means, km = k_means_clustering(svd_dataset, y_true_20, n_clusters=20, random_state=0, max_iter=1000, n_init=30)
    svd_homogeneity_1ist.append(k_means['homogeneity'])
    svd_completeness_1ist.append(k_means['completeness'])
    svd_v_measure_1ist.append(k_means['v_measure'])
    svd_adjusted_rand_index_1ist.append(k_means['adjusted_rand_index'])
    svd_adjusted_mutual_info_1ist.append(k_means['adjusted_mutual_info'])

plot_histogram('SVD Homogeneity Scores', svd_homogeneity_1ist)
plot_histogram('SVD Completeness Scores', svd_completeness_1ist)
plot_histogram('SVD V-Measure Scores', svd_v_measure_1ist)
plot_histogram('SVD Adjusted Rand Index Scores', svd_adjusted_rand_index_1ist)
plot_histogram('SVD Adjusted Mutual Information Scores', svd_adjusted_mutual_info_1ist)


# Question 9: NMF (Frobenius)
nmf_homogeneity_1ist = []
nmf_completeness_1ist = []
nmf_v_measure_1ist = []
nmf_adjusted_rand_index_1ist = []
nmf_adjusted_mutual_info_1ist = []

for r in [1, 2, 3, 5, 10, 20, 50, 100, 300]:
    print('n_components:',r)
    nmf_dataset, nmf = nmf_dimension_reduction(X_train_tfidf_20, n_components = r, solver='mu', beta_loss='frobenius', max_iter=1000, random_state=0)
    k_means, km = k_means_clustering(nmf_dataset, y_true_20, n_clusters=20, random_state=0, max_iter=1000, n_init=30)
    nmf_homogeneity_1ist.append(k_means['homogeneity'])
    nmf_completeness_1ist.append(k_means['completeness'])
    nmf_v_measure_1ist.append(k_means['v_measure'])
    nmf_adjusted_rand_index_1ist.append(k_means['adjusted_rand_index'])
    nmf_adjusted_mutual_info_1ist.append(k_means['adjusted_mutual_info'])

plot_histogram('NMF Frobenius Homogeneity Scores', nmf_homogeneity_1ist)
plot_histogram('NMF Frobenius Completeness Scores', nmf_completeness_1ist)
plot_histogram('NMF Frobenius V-Measure Scores', nmf_v_measure_1ist)
plot_histogram('NMF Frobenius Adjusted Rand Index Scores', nmf_adjusted_rand_index_1ist)
plot_histogram('NMF Frobenius Adjusted Mutual Information Scores', nmf_adjusted_mutual_info_1ist)


# Question 10: NMF 'Kullback-Leibler'
nmf_kl_homogeneity_1ist = []
nmf_kl_completeness_1ist = []
nmf_kl_v_measure_1ist = []
nmf_kl_adjusted_rand_index_1ist = []
nmf_kl_adjusted_mutual_info_1ist = []

for r in [1, 2, 3, 5, 10, 20, 50, 100, 300]:
    print('n_components:',r)
    nmf_dataset, nmf = nmf_dimension_reduction(X_train_tfidf_20, n_components = r, solver='mu', beta_loss='kullback-leibler', max_iter=1000, random_state=0)
    k_means, km = k_means_clustering(nmf_dataset, y_true_20, n_clusters=20, random_state=0, max_iter=1000, n_init=30)
    nmf_kl_homogeneity_1ist.append(k_means['homogeneity'])
    nmf_kl_completeness_1ist.append(k_means['completeness'])
    nmf_kl_v_measure_1ist.append(k_means['v_measure'])
    nmf_kl_adjusted_rand_index_1ist.append(k_means['adjusted_rand_index'])
    nmf_kl_adjusted_mutual_info_1ist.append(k_means['adjusted_mutual_info'])

plot_histogram('NMF Kullback-Leibler Homogeneity Scores', nmf_kl_homogeneity_1ist)
plot_histogram('NMF Kullback-Leibler Completeness Scores', nmf_kl_completeness_1ist)
plot_histogram('NMF Kullback-Leibler V-Measure Scores', nmf_kl_v_measure_1ist)
plot_histogram('NMF Kullback-Leibler Adjusted Rand Index Scores', nmf_kl_adjusted_rand_index_1ist)
plot_histogram('NMF Kullback-Leibler Adjusted Mutual Information Scores', nmf_kl_adjusted_mutual_info_1ist)


# Question 11: UMAP 'Euclidean'
umap_euclidean_homogeneity_1ist = []
umap_euclidean_completeness_1ist = []
umap_euclidean_v_measure_1ist = []
umap_euclidean_adjusted_rand_index_1ist = []
umap_euclidean_adjusted_mutual_info_1ist = []

for r in [1, 2, 3, 5, 10, 20, 50, 100, 300]:
    print('n_components:',r)
    umap_dataset, umap = umap_dimension_reduction(X_train_tfidf_20, n_components = r, metric='euclidean', disconnection_distance=2, random_state=0)
    k_means, km = k_means_clustering(umap_dataset, y_true_20, n_clusters=20, random_state=0, max_iter=1000, n_init=30)
    umap_euclidean_homogeneity_1ist.append(k_means['homogeneity'])
    umap_euclidean_completeness_1ist.append(k_means['completeness'])
    umap_euclidean_v_measure_1ist.append(k_means['v_measure'])
    umap_euclidean_adjusted_rand_index_1ist.append(k_means['adjusted_rand_index'])
    umap_euclidean_adjusted_mutual_info_1ist.append(k_means['adjusted_mutual_info'])

plot_histogram('UMAP Euclidean Homogeneity Scores', umap_euclidean_homogeneity_1ist)
plot_histogram('UMAP Euclidean Completeness Scores', umap_euclidean_completeness_1ist)
plot_histogram('UMAP Euclidean V-Measure Scores', umap_euclidean_v_measure_1ist)
plot_histogram('UMAP Euclidean Adjusted Rand Index Scores', umap_euclidean_adjusted_rand_index_1ist)
plot_histogram('UMAP Euclidean Adjusted Mutual Information Scores', umap_euclidean_adjusted_mutual_info_1ist)


# Question 11: UMAP 'Cosine'
umap_cosine_homogeneity_1ist = []
umap_cosine_completeness_1ist = []
umap_cosine_v_measure_1ist = []
umap_cosine_adjusted_rand_index_1ist = []
umap_cosine_adjusted_mutual_info_1ist = []

for r in [1, 2, 3, 5, 10, 20, 50, 100, 300]:
    print('n_components:',r)
    umap_dataset, umap = umap_dimension_reduction(X_train_tfidf_20, n_components=r, metric='cosine', disconnection_distance=2, random_state=0)
    k_means, km = k_means_clustering(umap_dataset, y_true_20, n_clusters=20, random_state=0, max_iter=1000, n_init=30)
    umap_cosine_homogeneity_1ist.append(k_means['homogeneity'])
    umap_cosine_completeness_1ist.append(k_means['completeness'])
    umap_cosine_v_measure_1ist.append(k_means['v_measure'])
    umap_cosine_adjusted_rand_index_1ist.append(k_means['adjusted_rand_index'])
    umap_cosine_adjusted_mutual_info_1ist.append(k_means['adjusted_mutual_info'])

plot_histogram('UMAP Cosine Homogeneity Scores', umap_cosine_homogeneity_1ist)
plot_histogram('UMAP Cosine Completeness Scores', umap_cosine_completeness_1ist)
plot_histogram('UMAP Cosine V-Measure Scores', umap_cosine_v_measure_1ist)
plot_histogram('UMAP Cosine Adjusted Rand Index Scores', umap_cosine_adjusted_rand_index_1ist)
plot_histogram('UMAP Cosine Adjusted Mutual Information Scores', umap_cosine_adjusted_mutual_info_1ist)
