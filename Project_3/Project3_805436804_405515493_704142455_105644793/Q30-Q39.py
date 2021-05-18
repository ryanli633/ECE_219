import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.model_selection import KFold
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.prediction_algorithms.matrix_factorization import SVD

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error

# filepaths for dataset (UPDATE THESE TO WHERE YOU HAVE PLACED THE DATASETS IN YOUR DIRECTORIES)
# filepath for ratings.csv
ratings_file_path = "/Users/ryanli/Documents/ECE_219/workspace/Project_3/ml-latest-small/ratings.csv"

reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1, rating_scale=(0.5, 5))
data = Dataset.load_from_file(ratings_file_path, reader=reader)


#30
ref = {}
for j in data.raw_ratings:
    if j[0] in ref.keys():
        ref[j[0]].append(j[2])
    else:
        ref[j[0]] = []
        ref[j[0]].append(j[2])

user = {}
for j in ref.keys():
    user[j] = np.mean(ref[j])

rmse = 0
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    pred = [user[i[0]] for i in testset]
    true = [i[2] for i in testset]
    rmse += np.sqrt(mean_squared_error(true,pred))
avg_rmse = rmse/10.0

print("Average RMSE for naive collaborative filter: ", avg_rmse)


#31
ref = {}
for j in data.raw_ratings:
    if j[0] in ref.keys():
        ref[j[0]].append(j[2])
    else:
        ref[j[0]] = []
        ref[j[0]].append(j[2])

user = {}
for j in ref.keys():
    user[j] = np.mean(ref[j])

ref1 = {}
for j in data.raw_ratings:
    if j[1] in ref1.keys():
        ref1[j[1]].append(j[2])
    else:
        ref1[j[1]] = []
        ref1[j[1]].append(j[2])

rmse = 0
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    pop_trim = [j for j in testset if len(ref1[j[1]]) > 2]
    pred = [user[i[0]] for i in pop_trim]
    true = [i[2] for i in pop_trim]
    rmse += np.sqrt(mean_squared_error(true,pred))
avg_rmse = rmse/10.0

print("Average RMSE for naive collaborative filter with Popular Movie Trimming: ", avg_rmse)


#32
ref = {}
for j in data.raw_ratings:
    if j[0] in ref.keys():
        ref[j[0]].append(j[2])
    else:
        ref[j[0]] = []
        ref[j[0]].append(j[2])

user = {}
for j in ref.keys():
    user[j] = np.mean(ref[j])

ref1 = {}
for j in data.raw_ratings:
    if j[1] in ref1.keys():
        ref1[j[1]].append(j[2])
    else:
        ref1[j[1]] = []
        ref1[j[1]].append(j[2])

rmse = 0
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    unpop_trim = [j for j in testset if len(ref1[j[1]]) <= 2]
    pred = [user[i[0]] for i in unpop_trim]
    true = [i[2] for i in unpop_trim]
    rmse += np.sqrt(mean_squared_error(true,pred))
avg_rmse = rmse/10.0

print("Average RMSE for naive collaborative filter with Unpopular Movie Trimming: ", avg_rmse)


#33
ref = {}
for j in data.raw_ratings:
    if j[0] in ref.keys():
        ref[j[0]].append(j[2])
    else:
        ref[j[0]] = []
        ref[j[0]].append(j[2])

user = {}
for j in ref.keys():
    user[j] = np.mean(ref[j])

ref1 = {}
for j in data.raw_ratings:
    if j[1] in ref1.keys():
        ref1[j[1]].append(j[2])
    else:
        ref1[j[1]] = []
        ref1[j[1]].append(j[2])

rmse = 0
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    highvar_trim = [j for j in testset if (len(ref1[j[1]]) >= 5 and np.var(ref1[j[1]]) >= 2)]
    pred = [user[i[0]] for i in highvar_trim]
    true = [i[2] for i in highvar_trim]
    rmse += np.sqrt(mean_squared_error(true,pred))
avg_rmse = rmse/10.0

print("Average RMSE for naive collaborative filter with High Variance Movie Trimming: ", avg_rmse)


#Q34
''' Plot the ROC curves (threshold = 3) for the k-NN, NNMF, and MF with bias
based collaborative filters in the same figure. Use the figure to compare the performance of the
filters in predicting the ratings of the movies.
'''
trainset, testset = train_test_split(data, test_size=.1)

pred = SVD(n_factors=22,verbose=False).fit(trainset).test(testset)
y_true = []
thresh = 3
for i in pred:
    if i.r_ui < thresh:
        y_true.append(0)
    else:
        y_true.append(1)

y_score = [i.est for i in pred]
fpr = dict();tpr = dict();roc_auc = dict()
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
lw = 2; plt.figure()
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='MF with bias (area = %0.4f)' % roc_auc)

pred = NMF(n_factors=18,verbose=False).fit(trainset).test(testset)
y_true = []
for i in pred:
    if i.r_ui < thresh:
        y_true.append(0)
    else:
        y_true.append(1)

y_score = [i.est for i in pred]
fpr = dict();tpr = dict();roc_auc = dict()
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
lw = 2
plt.plot(fpr, tpr, color='green',lw=lw, label='NMF (area = %0.4f)' % roc_auc)

pred = KNNWithMeans(k=20,sim_options={'name':'pearson'},verbose=False).fit(trainset).test(testset)
y_true = []
for i in pred:
    if i.r_ui < thresh:
        y_true.append(0)
    else:
        y_true.append(1)

y_score = [i.est for i in pred]
fpr = dict();tpr = dict();roc_auc = dict()
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
lw = 2
plt.plot(fpr, tpr, color='black',lw=lw, label='k-NN (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');
plt.title('ROC curve for threshold = %0.2f' % thresh);plt.legend(loc="lower right")
plt.show()


#36
t = np.linspace(1,25,num=25,dtype=int)
avg_precision_knn = []
avg_recall_knn = []
kf = KFold(n_splits=10)
for i in t:
    print(i)
    precision_fold = []
    recall_fold = []
    for trainset, testset in kf.split(data):
        G = {}
        for j in testset:
            if j[0] in G.keys():
                if j[2] >= 3.0:
                    G[j[0]].add(j[1])
            else:
                G[j[0]] = set()
                if j[2] >= 3.0:
                    G[j[0]].add(j[1])
        G_items = {}
        for j in testset:
            if j[0] in G_items.keys():
                G_items[j[0]].append(j[1])
            else:
                G_items[j[0]] = []
                G_items[j[0]].append(j[1])
        G_test = [j for j in testset if (len(G[j[0]]) > 0 and len(G_items[j[0]]) >= i)]
        pred = KNNWithMeans(k=20,sim_options={'name':'pearson'},verbose=False).fit(trainset).test(G_test)
        user = {}
        for u in pred:
            if u[0] in user.keys():
                item = (u[1],u[3])
                user[u[0]].append(item)
            else:
                user[u[0]] = []
                item = (u[1],u[3])
                user[u[0]].append(item)
        precision_user = []
        recall_user = []
        for u in user.keys():
            S = user[u]
            S = sorted(S,key=lambda x:x[1],reverse=True)
            S = S[:i]
            S_t = set([j[0] for j in S])
            G_truth = G[u]
            precision = len(S_t.intersection(G_truth))/float(len(S_t))
            recall = len(S_t.intersection(G_truth))/float(len(G_truth))
            precision_user.append(precision)
            recall_user.append(recall)
        precision_fold.append(np.mean(precision_user))
        recall_fold.append(np.mean(recall_user))
    avg_precision_knn.append(np.mean(precision_fold))
    avg_recall_knn.append(np.mean(recall_fold))
    print(avg_precision_knn)
    print(avg_recall_knn)

fig, ax = plt.subplots()
ax.plot(t,avg_precision_knn, 'r', label='Average Precision')
ax.legend(loc='best')
plt.xlabel("t"); plt.ylabel("Average Precision"); plt.title("k-NN collaborative filter (KNNWithMeans) with 10-fold CV")
plt.show()

fig, ax = plt.subplots()
ax.plot(t,avg_recall_knn, 'r', label='Average Recall')
ax.legend(loc='best')
plt.xlabel("t"); plt.ylabel("Average Recall"); plt.title("k-NN collaborative filter (KNNWithMeans) with 10-fold CV")
plt.show()

fig, ax = plt.subplots()
ax.plot(avg_recall_knn,avg_precision_knn, 'r')
ax.legend(loc='best')
plt.xlabel("Average Recall"); plt.ylabel("Average Precision"); plt.title("k-NN collaborative filter (KNNWithMeans) with 10-fold CV")
plt.show()


#37
t = np.linspace(1,25,num=25,dtype=int)
avg_precision_nmf = []
avg_recall_nmf = []
kf = KFold(n_splits=10)
for i in t:
    print(i)
    precision_fold = []
    recall_fold = []
    for trainset, testset in kf.split(data):
        G = {}
        for j in testset:
            if j[0] in G.keys():
                if j[2] >= 3.0:
                    G[j[0]].add(j[1])
            else:
                G[j[0]] = set()
                if j[2] >= 3.0:
                    G[j[0]].add(j[1])
        G_items = {}
        for j in testset:
            if j[0] in G_items.keys():
                G_items[j[0]].append(j[1])
            else:
                G_items[j[0]] = []
                G_items[j[0]].append(j[1])
        G_test = [j for j in testset if (len(G[j[0]]) > 0 and len(G_items[j[0]]) >= i)]
        pred = NMF(n_factors=20,verbose=False).fit(trainset).test(G_test)
        user = {}
        for u in pred:
            if u[0] in user.keys():
                item = (u[1],u[3])
                user[u[0]].append(item)
            else:
                user[u[0]] = []
                item = (u[1],u[3])
                user[u[0]].append(item)
        precision_user = []
        recall_user = []
        for u in user.keys():
            S = user[u]
            S = sorted(S,key=lambda x:x[1],reverse=True)
            S = S[:i]
            S_t = set([j[0] for j in S])
            G_truth = G[u]
            precision = len(S_t.intersection(G_truth))/float(len(S_t))
            recall = len(S_t.intersection(G_truth))/float(len(G_truth))
            precision_user.append(precision)
            recall_user.append(recall)
        precision_fold.append(np.mean(precision_user))
        recall_fold.append(np.mean(recall_user))
    avg_precision_nmf.append(np.mean(precision_fold))
    avg_recall_nmf.append(np.mean(recall_fold))
    print(avg_precision_nmf)
    print(avg_recall_nmf)

fig, ax = plt.subplots()
ax.plot(t,avg_precision_nmf, 'r', label='Average Precision')
ax.legend(loc='best')
plt.xlabel("t"); plt.ylabel("Average Precision"); plt.title("NMF collaborative filter with 10-fold CV")
plt.show()

fig, ax = plt.subplots()
ax.plot(t,avg_recall_nmf, 'r', label='Average Recall')
ax.legend(loc='best')
plt.xlabel("t"); plt.ylabel("Average Recall"); plt.title("NMF collaborative filter with 10-fold CV")
plt.show()

fig, ax = plt.subplots()
ax.plot(avg_recall_nmf,avg_precision_nmf, 'r')
ax.legend(loc='best')
plt.xlabel("Average Recall"); plt.ylabel("Average Precision"); plt.title("NMF collaborative filter with 10-fold CV")
plt.show()


#38
t = np.linspace(1,25,num=25,dtype=int)
avg_precision_svd = []
avg_recall_svd = []
kf = KFold(n_splits=10)
for i in t:
    print(i)
    precision_fold = []
    recall_fold = []
    for trainset, testset in kf.split(data):
        G = {}
        for j in testset:
            if j[0] in G.keys():
                if j[2] >= 3.0:
                    G[j[0]].add(j[1])
            else:
                G[j[0]] = set()
                if j[2] >= 3.0:
                    G[j[0]].add(j[1])
        G_items = {}
        for j in testset:
            if j[0] in G_items.keys():
                G_items[j[0]].append(j[1])
            else:
                G_items[j[0]] = []
                G_items[j[0]].append(j[1])
        G_test = [j for j in testset if (len(G[j[0]]) > 0 and len(G_items[j[0]]) >= i)]
        pred = SVD(n_factors=18,verbose=False).fit(trainset).test(G_test)
        user = {}
        for u in pred:
            if u[0] in user.keys():
                item = (u[1],u[3])
                user[u[0]].append(item)
            else:
                user[u[0]] = []
                item = (u[1],u[3])
                user[u[0]].append(item)
        precision_user = []
        recall_user = []
        for u in user.keys():
            S = user[u]
            S = sorted(S,key=lambda x:x[1],reverse=True)
            S = S[:i]
            S_t = set([j[0] for j in S])
            G_truth = G[u]
            precision = len(S_t.intersection(G_truth))/float(len(S_t))
            recall = len(S_t.intersection(G_truth))/float(len(G_truth))
            precision_user.append(precision)
            recall_user.append(recall)
        precision_fold.append(np.mean(precision_user))
        recall_fold.append(np.mean(recall_user))
    avg_precision_svd.append(np.mean(precision_fold))
    avg_recall_svd.append(np.mean(recall_fold))
    print(avg_precision_svd)
    print(avg_recall_svd)

fig, ax = plt.subplots()
ax.plot(t,avg_precision_svd, 'r', label='Average Precision')
ax.legend(loc='best')
plt.xlabel("t"); plt.ylabel("Average Precision"); plt.title("MF with bias collaborative filter with 10-fold CV")
plt.show()

fig, ax = plt.subplots()
ax.plot(t,avg_recall_svd, 'r', label='Average Recall')
ax.legend(loc='best')
plt.xlabel("t"); plt.ylabel("Average Recall"); plt.title("MF with bias collaborative filter with 10-fold CV")
plt.show()

fig, ax = plt.subplots()
ax.plot(avg_recall_svd,avg_precision_svd, 'r')
ax.legend(loc='best')
plt.xlabel("Average Recall"); plt.ylabel("Average Precision"); plt.title("MF with bias collaborative filter with 10-fold CV")
plt.show()


#39
plt.figure()
plt.plot(avg_recall_svd, avg_precision_svd, color='darkorange',lw=2, label='MF with bias')
plt.plot(avg_recall_nmf, avg_precision_nmf, color='green',lw=2, label='NMF')
plt.plot(avg_recall_knn, avg_precision_knn, color='black',lw=2, label='k-NN')
plt.xlabel("Average Recall"); plt.ylabel("Average Precision"); plt.title("Comparision of k-NN, NMF and MF with bias")
plt.legend(loc="best")
plt.show()
