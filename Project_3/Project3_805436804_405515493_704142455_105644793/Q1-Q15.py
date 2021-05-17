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

from sklearn.metrics import roc_curve, auc

# filepaths for dataset (UPDATE THESE TO WHERE YOU HAVE PLACED THE DATASETS IN YOUR DIRECTORIES)
# filepath for ratings.csv
ratings_file_path = "/Users/ryanli/Documents/ECE_219/workspace/Project_3/ml-latest-small/ratings.csv"


#1
ratings = pd.read_csv(ratings_file_path)

user_id = ratings['userId'].values
movie_id = ratings['movieId'].values
rating = ratings['rating'].values

sparsity = len(ratings)/float(len(set(movie_id))*len(set(user_id)))
print("Sparsity: ", sparsity)


#2
bins = np.linspace(0,5,num=11)
density, bins, _ = plt.hist(rating,bins=bins,edgecolor="black")
count, _ = np.histogram(rating, bins)
plt.xlabel("Rating score"); plt.ylabel("Frequency"); plt.title("Frequency of rating values")
for x,y,num in zip(bins, density, count):
    if num != 0:
        plt.text(x, y+250, num)
plt.show()


#3
counter = Counter(movie_id)
num_ratings = sorted(list(counter.values()),reverse=True)
plt.plot(num_ratings)
plt.xlabel("Movie index"); plt.ylabel("Number of ratings"); plt.title("Number of ratings for each movie index")
plt.show()


#4
counter = Counter(user_id)
num_users = sorted(list(counter.values()),reverse=True)
plt.plot(num_users)
plt.xlabel("User index"); plt.ylabel("Number of ratings"); plt.title("Number of ratings for each user index")
plt.show()


#6
reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1, rating_scale=(0.5, 5))
data = Dataset.load_from_file(ratings_file_path, reader=reader)

ref = {}
for j in data.raw_ratings:
    if j[1] in ref.keys():
        ref[j[1]].append(j[2])
    else:
        ref[j[1]] = []
        ref[j[1]].append(j[2])

var = {}
for i in ref.keys():
    var[i] = np.var(ref[i])

var_val = list(var.values())
bins = np.linspace(0,5,num=11)
plt.figure()
plt.hist(var_val,bins=bins,edgecolor="black")
plt.xlabel("Rating variance"); plt.ylabel("Number of movies"); plt.title("Frequency of rating variance")
plt.show()


#10
reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1, rating_scale=(0.5, 5))
data = Dataset.load_from_file(ratings_file_path, reader=reader)

avg_rmse = []
avg_mae = []
k = np.linspace(2,100,num=50,dtype=int)
for i in k:
    print(i)
    perf = cross_validate(KNNWithMeans(k=i,sim_options={'name':'pearson'},random_state=100),data,cv=10,measures=['rmse','mae'])
    avg_rmse.append(np.mean(perf['test_rmse']))
    avg_mae.append(np.mean(perf['test_mae']))

fig, ax = plt.subplots()
ax.plot(k,avg_rmse, 'r', label='Average RMSE')
ax.plot(k, avg_mae, 'b', label='Average MAE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("k-NN collaborative filter (KNNWithMeans) with 10-fold CV")
print(avg_rmse)
print(avg_mae)


#12
kf = KFold(n_splits=10,random_state=0)
k = np.linspace(2,100,num=50,dtype=int)
avg_rmse_pop_trim = []
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
        pred = KNNWithMeans(k=i,sim_options={'name':'pearson'},verbose=False).fit(trainset).test(pop_trim)
        rmse += accuracy.rmse(pred,verbose=False)
    avg_rmse_pop_trim.append(rmse/10.0)

print(avg_rmse_pop_trim)
print("Minimum average RMSE for Popular Movie Trimming: ", min(avg_rmse_pop_trim), " at k = ", (avg_rmse_pop_trim.index(min(avg_rmse_pop_trim))+1)*2)
fig, ax = plt.subplots()
ax.plot(k,avg_rmse_pop_trim, 'r', label='Average RMSE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("k-NN collaborative filter (KNNWithMeans) with 10-fold CV on Popular Movie Trimming")
plt.show()


#13
kf = KFold(n_splits=10)
k = np.linspace(2,100,num=50,dtype=int)
avg_rmse_unpop_trim = []
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
        pred = KNNWithMeans(k=i,sim_options={'name':'pearson'},verbose=False).fit(trainset).test(unpop_trim)
        rmse += accuracy.rmse(pred,verbose=False)
    avg_rmse_unpop_trim.append(rmse/10.0)

print(avg_rmse_unpop_trim)
print("Minimum average RMSE for Unpopular Movie Trimming: ", min(avg_rmse_unpop_trim), " at k = ", (avg_rmse_unpop_trim.index(min(avg_rmse_unpop_trim))+1)*2)
fig, ax = plt.subplots()
ax.plot(k,avg_rmse_unpop_trim, 'r', label='Average RMSE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("k-NN collaborative filter (KNNWithMeans) with 10-fold CV on Unpopular Movie Trimming")
plt.show()


#14
kf = KFold(n_splits=10)
k = np.linspace(2,100,num=50,dtype=int)
avg_rmse_highvar_trim = []
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
        pred = KNNWithMeans(k=i,sim_options={'name':'pearson'},verbose=False).fit(trainset).test(highvar_trim)
        rmse += accuracy.rmse(pred,verbose=False)
    avg_rmse_highvar_trim.append(rmse/10.0)

print(avg_rmse_highvar_trim)
print("Minimum average RMSE for High variance Movie Trimming: ", min(avg_rmse_highvar_trim), " at k = ", (avg_rmse_highvar_trim.index(min(avg_rmse_highvar_trim))+1)*2)
fig, ax = plt.subplots()
ax.plot(k,avg_rmse_highvar_trim, 'r', label='Average RMSE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("k-NN collaborative filter (KNNWithMeans) with 10-fold CV on High Variance Movie Trimming")
plt.show()


#15
trainset, testset = train_test_split(data, test_size=.1,random_state=100)
pred = KNNWithMeans(k=22,sim_options={'name':'pearson','user_based':True},random_state=100,verbose=False).fit(trainset).test(testset)
y_true = []

threshes = [2.5, 3.0, 3.5, 4.0]
for thresh in threshes:
    y_true=[]
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
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');
    plt.title('ROC curve for threshold = %0.2f' % thresh);plt.legend(loc="lower right")
    plt.show()
