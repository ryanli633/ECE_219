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

# filepaths for dataset (UPDATE THESE TO WHERE YOU HAVE PLACED THE DATASETS IN YOUR DIRECTORIES)
# filepath for ratings.csv
ratings_file_path = "/Users/ryanli/Documents/ECE_219/workspace/Project_3/ml-latest-small/ratings.csv"

reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1, rating_scale=(0.5, 5))
data = Dataset.load_from_file(ratings_file_path, reader=reader)


#Q24
'''Design a MF with bias collaborative filter to predict the ratings of the movies
in the MovieLens dataset and evaluate it’s performance using 10-fold cross-validation. Sweep k
(number of latent factors) from 2 to 50 in step sizes of 2, and for each k compute the average
RMSE and average MAE obtained by averaging the RMSE and MAE across all 10 folds. Plot the
average RMSE (Y-axis) against k (X-axis) and the average MAE (Y-axis) against k (X-axis). For
solving this question, use the default value for the regularization parameter.'''

avg_rmse = []
avg_mae = []
k = np.linspace(2,50,num=25,dtype=int)
for i in k:
    print(i)
    perf = cross_validate(SVD(n_factors=i,verbose=False),data,cv=10)
    avg_rmse.append(np.mean(perf['test_rmse']))
    avg_mae.append(np.mean(perf['test_mae']))

print("Minimum average RMSE: ", min(avg_rmse))
print("Minimum average MAE: ", min(avg_mae))
fig, ax = plt.subplots()
ax.plot(k,avg_rmse, 'r', label='Average RMSE')
ax.plot(k, avg_mae, 'b', label='Average MAE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("MF with bias collaborative filter with 10-fold CV")
plt.show()

print("Minimum average RMSE: ", min(avg_rmse))
print("Minimum average MAE: ", min(avg_mae))
fig, ax = plt.subplots()
ax.plot(k,avg_rmse, 'r', label='Average RMSE')
#ax.plot(k, avg_mae, 'b', label='Average MAE')
ax.set_ylabel("Average RMSE",color="red",fontsize=14)
ax.set_xlabel("k",fontsize=14)

ax2=ax.twinx()
ax2.plot(k, avg_mae, 'b', label='Average MAE')
ax2.set_ylabel("Average MAE",color="blue",fontsize=14)

plt.title("MF with bias collaborative filter with 10-fold CV")
plt.show()


#26
kf = KFold(n_splits=10)
k = np.linspace(2,50,num=25,dtype=int)
avg_rmse = []
ref = {}
for j in data.raw_ratings:
    if j[1] in ref.keys():
        ref[j[1]].append(j[2])
    else:
        ref[j[1]] = []
        ref[j[1]].append(j[2])
        
for i in k:
    print(i)
    rmse = 0
    for trainset, testset in kf.split(data):
        pop_trim = [j for j in testset if len(ref[j[1]]) > 2]
        pred = SVD(n_factors=i,verbose=False).fit(trainset).test(pop_trim)
        rmse += accuracy.rmse(pred,verbose=False)
    avg_rmse.append(rmse/10.0)

print("Minimum average RMSE for Popular Movie Trimming: ", min(avg_rmse))
fig, ax = plt.subplots()
ax.plot(k,avg_rmse, 'r', label='Average RMSE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("MF with bias collaborative filter with 10-fold CV on Popular Movie Trimming")
plt.show()


#27
kf = KFold(n_splits=10)
k = np.linspace(2,50,num=25,dtype=int)
avg_rmse = []
ref = {}
for j in data.raw_ratings:
    if j[1] in ref.keys():
        ref[j[1]].append(j[2])
    else:
        ref[j[1]] = []
        ref[j[1]].append(j[2])

for i in k:
    print(i)
    rmse = 0
    for trainset, testset in kf.split(data):
        unpop_trim = [j for j in testset if len(ref[j[1]]) <= 2]
        pred = SVD(n_factors = i,verbose=False).fit(trainset).test(unpop_trim)
        rmse += accuracy.rmse(pred,verbose=False)
    avg_rmse.append(rmse/10.0)

print("Minimum average RMSE for Unpopular Movie Trimming: ", min(avg_rmse))
fig, ax = plt.subplots()
ax.plot(k,avg_rmse, 'r', label='Average RMSE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("MF with bias collaborative filter with 10-fold CV on Unpopular Movie Trimming")
plt.show()


#28
kf = KFold(n_splits=10)
k = np.linspace(2,50,num=25,dtype=int)
avg_rmse = []
ref = {}
for j in data.raw_ratings:
    if j[1] in ref.keys():
        ref[j[1]].append(j[2])
    else:
        ref[j[1]] = []
        ref[j[1]].append(j[2])
        
for i in k:
    print(i)
    rmse = 0
    for trainset, testset in kf.split(data):        
        highvar_trim = [j for j in testset if (len(ref[j[1]]) >= 5 and np.var(ref[j[1]]) >= 2)]
        pred = SVD(n_factors=i,verbose=False).fit(trainset).test(highvar_trim)
        rmse += accuracy.rmse(pred,verbose=False)
    avg_rmse.append(rmse/10.0)

print("Minimum average RMSE for High variance Movie Trimming: ", min(avg_rmse))
fig, ax = plt.subplots()
ax.plot(k,avg_rmse, 'r', label='Average RMSE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("MF with bias collaborative filter with 10-fold CV on High Variance Movie Trimming")
plt.show()


#Q29
''' Plot the ROC curves for the MF with bias collaborative filter designed in question
24 for threshold values [2.5, 3, 3.5, 4]. For the ROC plotting use the optimal number of latent factors
found in question 25. For each of the plots, also report the area under the curve (AUC) value.
'''

trainset, testset = train_test_split(data, test_size=.1)
pred = SVD(n_factors=22,verbose=False).fit(trainset).test(testset)
y_true = []
thresh = 2.5
for i in pred:
    if i.r_ui < thresh:
        y_true.append(0)
    else:
        y_true.append(1)

y_score = [i.est for i in pred]
fpr = dict();tpr = dict();roc_auc = dict()
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.subplots(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14); plt.ylabel('True Positive Rate', fontsize=14);
plt.title('ROC curve for threshold = %0.2f' % thresh, fontsize=16);plt.legend(loc="lower right")
plt.show()


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
plt.figure()
lw = 2
plt.subplots(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14); plt.ylabel('True Positive Rate', fontsize=14);
plt.title('ROC curve for threshold = %0.2f' % thresh, fontsize=16);plt.legend(loc="lower right")
plt.show()


trainset, testset = train_test_split(data, test_size=.1)
pred = SVD(n_factors=22,verbose=False).fit(trainset).test(testset)
y_true = []
thresh = 3.5
for i in pred:
    if i.r_ui < thresh:
        y_true.append(0)
    else:
        y_true.append(1)

y_score = [i.est for i in pred]
fpr = dict();tpr = dict();roc_auc = dict()
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.subplots(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14); plt.ylabel('True Positive Rate', fontsize=14);
plt.title('ROC curve for threshold = %0.2f' % thresh, fontsize=16);plt.legend(loc="lower right")
plt.show()


trainset, testset = train_test_split(data, test_size=.1)
pred = SVD(n_factors=22,verbose=False).fit(trainset).test(testset)
y_true = []
thresh = 4
for i in pred:
    if i.r_ui < thresh:
        y_true.append(0)
    else:
        y_true.append(1)

y_score = [i.est for i in pred]
fpr = dict();tpr = dict();roc_auc = dict()
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.subplots(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14); plt.ylabel('True Positive Rate', fontsize=14);
plt.title('ROC curve for threshold = %0.2f' % thresh, fontsize=16);plt.legend(loc="lower right")
plt.show()
