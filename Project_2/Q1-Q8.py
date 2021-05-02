#!/usr/bin/env python
# coding: utf-8

import numpy as np
np.random.seed(42)
import random
random.seed(42)

import nltk, string, itertools
import matplotlib.pyplot as plt
import itertools

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from string import punctuation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF


def plot_contingency_table(cm, title='Contingency Table', cmap=plt.cm.YlOrBr,
                           actual_class_names=['Class 1', 'Class 2'],
                           cluster_class_names=['Cluster 1', 'Cluster 2']):
    plt.gcf().clear()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(actual_class_names))
    plt.xticks(tick_marks, actual_class_names)
    plt.yticks(tick_marks, cluster_class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Cluster Class', fontsize=12)
    plt.xlabel('Actual Class', fontsize=12)
    plt.show()


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


# Fetch dataset
categories = ['comp.sys.ibm.pc.hardware', 'comp.graphics','comp.sys.mac.hardware', 'comp.os.ms-windows.misc','rec.autos', 'rec.motorcycles','rec.sport.baseball', 'rec.sport.hockey']
dataset = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers'), shuffle=True, random_state=42)

# Custom stop words
stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(stop_words_skt))

# Question 1
tfidf_vect = TfidfVectorizer(stop_words=combined_stopwords,min_df=3)
X_train_tfidf = tfidf_vect.fit_transform(dataset.data) 
print("Shape of TF-IDF matrix: ", X_train_tfidf.shape)


# Question 2
y_true = [int(i/4) for i in dataset.target]

km = KMeans(n_clusters=2, random_state=0, max_iter=1000, n_init=30)
y_pred = km.fit_predict(X_train_tfidf)
con_mat = contingency_matrix(y_true,y_pred)
plot_contingency_table(con_mat)


# Question 3
print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))


# Question 4 
svd = TruncatedSVD(n_components=1000,random_state=0)
X_train_svd = svd.fit_transform(X_train_tfidf)
plt.figure()
plt.plot(np.arange(1000)+1,sorted(svd.explained_variance_ratio_,reverse=True))
plt.scatter(np.arange(1000)+1,sorted(svd.explained_variance_ratio_,reverse=True),)
plt.xlabel("Components"); plt.ylabel("Explained Variance Ratio per Component")

plt.figure()
plt.plot(np.arange(1000)+1,np.cumsum(sorted(svd.explained_variance_ratio_,reverse=True)))
plt.scatter(np.arange(1000)+1,np.cumsum(sorted(svd.explained_variance_ratio_,reverse=True)))
plt.xlabel("Components"); plt.ylabel("Total Explained Variance Ratio")


# Question 5: SVD
r = [1,2,3,5,10,20,50,100,300]
hom_score = []; complt_score = []; v_score = []; adj_rand_score = []; adj_mut_inf_score = []
for i in r:
    y_pred = km.fit_predict(TruncatedSVD(n_components=i,random_state=0).fit_transform(X_train_tfidf))
    hom_score.append(homogeneity_score(y_true,y_pred))
    complt_score.append(completeness_score(y_true,y_pred))
    v_score.append(v_measure_score(y_true,y_pred))
    adj_rand_score.append(adjusted_rand_score(y_true,y_pred))
    adj_mut_inf_score.append(adjusted_mutual_info_score(y_true,y_pred))

fig, ax = plt.subplots()
ax.plot(r,hom_score, 'r', label='Homogeneity score')
ax.plot(r, complt_score, 'b', label='Completeness score')
ax.plot(r, v_score, 'g', label='V-measure score')
ax.plot(r,adj_rand_score,'y',label='Adjusted Rand score')
ax.plot(r,adj_mut_inf_score,'m',label='Adjusted Mutual Information score')
ax.legend(loc='best')
plt.xlabel("Number of components"); plt.ylabel("Score"); plt.title("Measure scores for SVD")
print("SVD")
print(hom_score)
print(complt_score)
print(v_score)
print(adj_rand_score)
print(adj_mut_inf_score)

plot_histogram('SVD Homogeneity Scores', hom_score)
plot_histogram('SVD Completeness Scores', complt_score)
plot_histogram('SVD V-Measure Scores', v_score)
plot_histogram('SVD Adjusted Rand Index Scores', adj_rand_score)
plot_histogram('SVD Adjusted Mutual Information Scores', adj_mut_inf_score)


# Question 5: NMF
hom_score_nmf = []; complt_score_nmf = []; v_score_nmf = []; adj_rand_score_nmf = []; adj_mut_inf_score_nmf = []
for i in r:
    y_pred_nmf = km.fit_predict(NMF(n_components=i,init='random',random_state=0,max_iter=1000).fit_transform(X_train_tfidf))
    hom_score_nmf.append(homogeneity_score(y_true,y_pred_nmf))
    complt_score_nmf.append(completeness_score(y_true,y_pred_nmf))
    v_score_nmf.append(v_measure_score(y_true,y_pred_nmf))
    adj_rand_score_nmf.append(adjusted_rand_score(y_true,y_pred_nmf))
    adj_mut_inf_score_nmf.append(adjusted_mutual_info_score(y_true,y_pred_nmf))

fig, ax = plt.subplots()
ax.plot(r, hom_score_nmf, 'r', label='Homogeneity score')
ax.plot(r, complt_score_nmf, 'b', label='Completeness score')
ax.plot(r, v_score_nmf, 'g', label='V-measure score')
ax.plot(r, adj_rand_score_nmf,'y',label='Adjusted Rand Index')
ax.plot(r, adj_mut_inf_score_nmf,'m',label='Adjusted Mutual Information score')
ax.legend(loc='best')
plt.xlabel("Number of components"); plt.ylabel("Score"); plt.title("Measure score for NMF")
print("NMF")
print(hom_score_nmf)
print(complt_score_nmf)
print(v_score_nmf)
print(adj_rand_score_nmf)
print(adj_mut_inf_score_nmf)

plot_histogram('NMF Homogeneity Scores', hom_score_nmf)
plot_histogram('NMF Completeness Scores', complt_score_nmf)
plot_histogram('NMF V-Measure Scores', v_score_nmf)
plot_histogram('NMF Adjusted Rand Index Scores', adj_rand_score_nmf)
plot_histogram('NMF Adjusted Mutual Information Scores', adj_mut_inf_score_nmf)


# Question 7
r_best_svd = 300
r_best_nmf = 2

reduced_data_svd = TruncatedSVD(n_components=r_best_svd,random_state=0).fit_transform(X_train_tfidf)
plt.figure()
plt.scatter(reduced_data_svd[:,0],reduced_data_svd[:,1],c=y_true,cmap='viridis')
plt.title("SVD Ground truth class labels (r=300)");

km = KMeans(n_clusters=2, random_state=0, max_iter=1000, n_init=30)
svd_labels = km.fit_predict(reduced_data_svd)
plt.figure()
plt.scatter(reduced_data_svd[:,0],reduced_data_svd[:,1],c=svd_labels,cmap='viridis')
plt.title("SVD Clustering class labels (r=300)");

reduced_data_nmf = NMF(n_components=r_best_nmf,init='random',random_state=0).fit_transform(X_train_tfidf)
plt.figure()
plt.scatter(reduced_data_nmf[:,0],reduced_data_nmf[:,1],c=y_true,cmap='viridis')
plt.title("NMF Ground truth class labels (r=2)");

nmf_labels = km.fit_predict(reduced_data_nmf)
plt.figure()
plt.scatter(reduced_data_nmf[:,0],reduced_data_nmf[:,1],c=nmf_labels,cmap='viridis')
plt.title("NMF Clustering class labels (r=2)");
