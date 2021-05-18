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
# filepath for movies.csv
movies_file_path = "/Users/ryanli/Documents/ECE_219/workspace/Project_3/ml-latest-small/movies.csv"

reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1, rating_scale=(0.5, 5))
data = Dataset.load_from_file(ratings_file_path, reader=reader)


#17
avg_rmse_nmf = []
avg_mae_nmf = []
k = np.linspace(2,50,num=25,dtype=int)
for i in k:
    print(i)
    perf = cross_validate(NMF(n_factors=i,verbose=False,random_state=100),data,cv=10)
    avg_rmse_nmf.append(np.mean(perf['test_rmse']))
    avg_mae_nmf.append(np.mean(perf['test_mae']))

print("Minimum average RMSE: ", min(avg_rmse_nmf), " at k = ", (avg_rmse_nmf.index(min(avg_rmse_nmf))+1)*2)
print("Minimum average MAE: ", min(avg_mae_nmf), " at k = ", (avg_mae_nmf.index(min(avg_mae_nmf))+1)*2)
fig, ax = plt.subplots()
ax.plot(k,avg_rmse_nmf, 'r', label='Average RMSE')
ax.plot(k, avg_mae_nmf, 'b', label='Average MAE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("NMF collaborative filter with 10-fold CV")
plt.show()


#19
kf = KFold(n_splits=10,random_state=100)
k = np.linspace(2,50,num=25,dtype=int)
avg_rmse_nnmf_pop = []
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
        pred = NMF(n_factors=i,verbose=False,random_state=100).fit(trainset).test(pop_trim)
        rmse += accuracy.rmse(pred,verbose=False)
    avg_rmse_nnmf_pop.append(rmse/10.0)

print("Minimum average RMSE for Popular Movie Trimming: ", min(avg_rmse_nnmf_pop), " at k = ", (avg_rmse_nnmf_pop.index(min(avg_rmse_nnmf_pop))+1)*2)
fig, ax = plt.subplots()
ax.plot(k,avg_rmse_nnmf_pop, 'r', label='Average RMSE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("NMF collaborative filter with 10-fold CV on Popular Movie Trimming")
plt.show()


#20
kf = KFold(n_splits=10,random_state=100)
k = np.linspace(2,50,num=25,dtype=int)
avg_rmse_nnmf_unpop = []
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
        pred = NMF(n_factors = i,verbose=False,random_state=100).fit(trainset).test(unpop_trim)
        rmse += accuracy.rmse(pred,verbose=False)
    avg_rmse_nnmf_unpop.append(rmse/10.0)

print("Minimum average RMSE for Unpopular Movie Trimming: ", min(avg_rmse_nnmf_unpop), " at k = ", (avg_rmse_nnmf_unpop.index(min(avg_rmse_nnmf_unpop))+1)*2)
fig, ax = plt.subplots()
ax.plot(k,avg_rmse_nnmf_unpop, 'r', label='Average RMSE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("NMF collaborative filter with 10-fold CV on Unpopular Movie Trimming")
plt.show()


#21
kf = KFold(n_splits=10,random_state=100)
k = np.linspace(2,50,num=25,dtype=int)
avg_rmse_nnmf_highvar = []
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
        pred = NMF(n_factors=i,verbose=False,random_state=100).fit(trainset).test(highvar_trim)
        rmse += accuracy.rmse(pred,verbose=False)
    avg_rmse_nnmf_highvar.append(rmse/10.0)

print("Minimum average RMSE for High variance Movie Trimming: ", min(avg_rmse_nnmf_highvar), " at k = ", (avg_rmse_nnmf_highvar.index(min(avg_rmse_nnmf_highvar))+1)*2)
fig, ax = plt.subplots()
ax.plot(k,avg_rmse_nnmf_highvar, 'r', label='Average RMSE')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("NMF collaborative filter with 10-fold CV on High Variance Movie Trimming")
plt.show()


#22
trainset, testset = train_test_split(data, test_size=.1,random_state=100)
pred = NMF(n_factors=16,verbose=False,random_state=100).fit(trainset).test(testset)
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


#23
trainset, testset = train_test_split(data, test_size=.1,random_state=100)
nmf = NMF(n_factors=20,verbose=False,random_state=100)
nmf.fit(trainset).test(testset)
V = nmf.qi
k = [item for item in range(0,20)]
df = pd.read_csv(movies_file_path,names=['movieid','title','genres'],header=0)
for i in k:
    print(i)
    mov = V[:,i]
    mov1 = [(n,j) for n,j in enumerate(mov)]
    mov1.sort(key = lambda x:x[1], reverse=True)
    for a in mov1[:10]:
        print(df['genres'][a[0]])
