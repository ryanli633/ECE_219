import kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
from string import punctuation
import string
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pprint
pp = pprint.PrettyPrinter(indent=4)
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score
import itertools
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import hdbscan
import time

########################### Data Acquisition ###########################

kaggle.api.authenticate()
#!kaggle competitions download -c learn-ai-bbc

#train_bbc=pd.read_csv('C:/Users/ohass/Downloads/learn-ai-bbc/BBC News Train.csv', engine='python')
#test_bbc=pd.read_csv('C:/Users/ohass/Downloads/learn-ai-bbc/BBC News Test.csv', engine='python')

train_docs=train_bbc.Text
test_docs=test_bbc.Text
bbc_cats=train_bbc.Category

########################### Explore Dataset ###########################

unique_cats=list(np.unique(bbc_cats))
str_cats=map(str, unique_cats)

cats_list=bbc_cats.tolist() #list of categories 
cat_count=[]
for n in unique_cats:    #looping through raw cats list and counting unique cats
    count = cats_list.count(n)
    cat_count.append(count)

plt.figure(figsize=(10, 6))
plt.title('Number of articles in each category', fontsize=20)
plt.xticks(fontsize=12)
plt.xlabel('Categories', fontsize=16)
plt.ylabel('Number of docs', fontsize=16)
plt.axis([-1,5,0,450])
plt.bar(unique_cats, cat_count)

for x,y in zip(unique_cats,cat_count):
    label = "{:.0f}".format(y)
    plt.annotate(label,
                (x,y),
                textcoords="offset points",
                xytext=(0,14),
                ha='center',
                fontsize=10)
    
########################### Feature Engineering ###########################

stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(stop_words_skt))

cat_index=[]
for i in bbc_cats:
    if i == 'business': cat_index.append(0)
    if i == 'entertainment': cat_index.append(1)
    if i == 'politics': cat_index.append(2)
    if i == 'sport': cat_index.append(3)
    if i == 'tech': cat_index.append(4)
        
def penn_to_wordnet(ptag):
    tags = {'JJ':'a',
           'NN':'n',
           'VB':'v',
           'RB':'r'}
    try:
        return tags[ptag[:2]]
    except:
        return 'n'
    
#lemmatize single document
def lemfxn(doc):
    wnlem = nltk.wordnet.WordNetLemmatizer()
    lemmatize = []
    for word, tag in pos_tag(nltk.word_tokenize(doc)):
        #if(not number_term(word)):  #removing number terms
        lemmword = wnlem.lemmatize(word.lower(), pos=penn_to_wordnet(tag))
        lemmatize.append(lemmword) 
    lem_output= ' '.join(lemmatize)
    return lem_output

#lemmetize set of docs
def lemmdata(doc):
    lemlist=[]
    for d in doc:
        lemdoc=lemfxn(d)  #lemm function
        lemlist.append(lemdoc)  #build lemmatized doc list
    return lemlist

vectorize = CountVectorizer(min_df=4, max_df=600, stop_words=combined_stopwords)
lemvectrain=vectorize.fit_transform(lemmdata(train_docs))
lemvectest=vectorize.transform(lemmdata(test_docs))

        
###################### 'english', min_df=3, No max_df 
tfidf_vect = TfidfVectorizer(stop_words='english', min_df=3)
X_train_tfidf = tfidf_vect.fit_transform(train_docs) 

km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = km.fit_predict(X_train_tfidf)
con_mat = contingency_matrix(y_true,y_pred)
pp.pprint(con_mat)

print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))

###################### 'combined_stopwords, min_df=3, No max_df 
tfidf_vect = TfidfVectorizer(stop_words=combined_stopwords, min_df=3)
X_train_tfidf = tfidf_vect.fit_transform(train_docs) 

km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = km.fit_predict(X_train_tfidf)
con_mat = contingency_matrix(y_true,y_pred)
pp.pprint(con_mat)

print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))

###################### 'english', lemmatized, min_df=3, No max_df 
tfidf_vect = TfidfVectorizer(stop_words='english', min_df=3)
X_train_tfidf = tfidf_vect.fit_transform(train_docs) 

km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = km.fit_predict(lemvectrain)
con_mat = contingency_matrix(y_true,y_pred)
pp.pprint(con_mat)

print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))

######################  combined_stopwords,lemmatized, min_df=3, No max_df 
tfidf_vect = TfidfVectorizer(stop_words=combined_stopwords, min_df=3)
X_train_tfidf = tfidf_vect.fit_transform(train_docs) 

km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = km.fit_predict(lemvectrain)
con_mat = contingency_matrix(y_true,y_pred)
pp.pprint(con_mat)

print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))

###################### 'english', min_df=5, No max_df 
tfidf_vect = TfidfVectorizer(stop_words='english', min_df=5)
X_train_tfidf = tfidf_vect.fit_transform(train_docs) 

km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = km.fit_predict(X_train_tfidf)
con_mat = contingency_matrix(y_true,y_pred)
pp.pprint(con_mat)

print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))

###################### 'english', min_df=4, No max_df 
tfidf_vect = TfidfVectorizer(stop_words='english', min_df=4)
X_train_tfidf = tfidf_vect.fit_transform(train_docs) 

km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = km.fit_predict(X_train_tfidf)
con_mat = contingency_matrix(y_true,y_pred)
pp.pprint(con_mat)

print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))

###################### 'english', min_df=4, max_df=1000 
tfidf_vect = TfidfVectorizer(stop_words='english', min_df=4, max_df=1000)
X_train_tfidf = tfidf_vect.fit_transform(train_docs) 

km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = km.fit_predict(X_train_tfidf)
con_mat = contingency_matrix(y_true,y_pred)
pp.pprint(con_mat)

print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))


###################### 'english', min_df=4, max_df=600 
tfidf_vect = TfidfVectorizer(stop_words='english', min_df=4, max_df=600)
X_train_tfidf = tfidf_vect.fit_transform(train_docs) 

km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = km.fit_predict(X_train_tfidf)
con_mat = contingency_matrix(y_true,y_pred)
pp.pprint(con_mat)

print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))

###################### combined_stopwords, min_df=4, max_df=600 
tfidf_vect = TfidfVectorizer(stop_words=combined_stopwords, min_df=4, max_df=600)
X_train_tfidf = tfidf_vect.fit_transform(train_docs)
X_test_tfidf = tfidf_vect.transform(test_docs)


km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = km.fit_predict(X_train_tfidf)
con_mat = contingency_matrix(y_true,y_pred)
pp.pprint(con_mat)

print("Homogeneity score: %0.3f" %homogeneity_score(y_true,y_pred))
print("Completeness score: %0.3f" %completeness_score(y_true,y_pred))
print("V-measure score: %0.3f" %v_measure_score(y_true,y_pred))
print("Adjusted Rand score: %0.3f" %adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: %0.3f" %adjusted_mutual_info_score(y_true,y_pred))

#Contingency table plot
def plot_contingency_table(cm, title='Contingency Table', cmap=plt.cm.YlOrBr,
                           actual_class_names=['business', 'entertainment', 'politics', 'sport', 'tech'],
                           cluster_class_names=['business', 'entertainment', 'politics', 'sport', 'tech']):
    plt.figure(figsize=(10, 6))
    plt.gcf().clear()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(actual_class_names))
    plt.xticks( tick_marks, actual_class_names, rotation=45)
    plt.yticks(tick_marks, cluster_class_names)

    thresh = cm.max() / 5.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Cluster Class', fontsize=16)
    plt.xlabel('Actual Class', fontsize=16)
    plt.show()
    
# reorder to maximize along diagonal
cm = confusion_matrix(y_true, y_pred)
rows, cols = linear_sum_assignment(cm, maximize=True)
new_cm = cm[rows[:,np.newaxis], cols]    
plot_contingency_table(new_cm)


########################### TruncatedSVD ###########################
svd = TruncatedSVD(n_components=1000,random_state=0)
X_train_svd = svd.fit_transform(X_train_tfidf)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1000)+1,sorted(svd.explained_variance_ratio_,reverse=True))
plt.scatter(np.arange(1000)+1,sorted(svd.explained_variance_ratio_,reverse=True),)
plt.xlabel("Components"); plt.ylabel("Explained Variance Ratio per Component")

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1000)+1,np.cumsum(sorted(svd.explained_variance_ratio_,reverse=True)))
plt.scatter(np.arange(1000)+1,np.cumsum(sorted(svd.explained_variance_ratio_,reverse=True)))
plt.xlabel("Components"); plt.ylabel("Total Explained Variance Ratio")

#Measure Scores for TruncatedSVD
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

########################### NMF ###########################
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

#SVD best r = 50
#NMF best r = 5

########################### Best SVD & NMF Scatter Clusters ###########################

r_best_svd = 50
r_best_nmf = 5

reduced_data_svd = TruncatedSVD(n_components=r_best_svd,random_state=0).fit_transform(X_train_tfidf)
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data_svd[:,0],reduced_data_svd[:,1],c=y_true,cmap='viridis')
plt.title("SVD Ground truth class labels (r= %i)" %r_best_svd);

km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=30)
svd_labels = km.fit_predict(reduced_data_svd)
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data_svd[:,0],reduced_data_svd[:,1],c=svd_labels,cmap='viridis')
plt.title("SVD Clustering class labels (r= %i)" %r_best_svd);

reduced_data_nmf = NMF(n_components=r_best_nmf,init='random',random_state=0).fit_transform(X_train_tfidf)
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data_nmf[:,0],reduced_data_nmf[:,1],c=y_true,cmap='viridis')
plt.title("NMF Ground truth class labels (r= %i)" %r_best_nmf);

nmf_labels = km.fit_predict(reduced_data_nmf)
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data_nmf[:,0],reduced_data_nmf[:,1],c=nmf_labels,cmap='viridis')
plt.title("NMF Clustering class labels (r= %i)" %r_best_nmf);

#NMF dimension reduction
def nmf_dimension_reduction(training_data, n_components=None, solver='cd', beta_loss='frobenius', max_iter=1000, random_state=0):
    start = time.time()
    nmf = NMF(n_components=n_components, solver=solver, init='nndsvda', beta_loss=beta_loss, max_iter=max_iter, random_state=random_state)
    nmf_dataset = nmf.fit_transform(training_data)
    print("NMF complete after %f seconds" % (time.time()-start))
    return nmf_dataset, nmf


########################### NMF with kullback-leibler ###########################

#Find best n_components for NMF with kullback-leibler
h_scores=[]
c_scores=[]
vm_scores=[]
ar_scores=[]
ami_scores=[]

for r in [1, 2, 3, 5, 10, 20, 50, 100, 300]:
    print('n_components:',r)
    nmf_dataset, nmf = nmf_dimension_reduction(X_train_tfidf, n_components = r, solver='mu', beta_loss='kullback-leibler', max_iter=1000, random_state=0)
    km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
    cat_index_a=np.array(cat_index)
    y_true = cat_index_a
    y_pred = km.fit_predict(nmf_dataset)
    con_mat = contingency_matrix(y_true,y_pred)
    pp.pprint(con_mat)
    
    cm = confusion_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(cm, maximize=True)
    new_cm = cm[rows[:,np.newaxis], cols]
    plot_contingency_table(new_cm, title= 'NMF with KL, n_components = %i' %r)
    print("Homogeneity score (n_components = %i): " %r, homogeneity_score(y_true,y_pred))
    print("Completeness score (n_components = %i): " %r, completeness_score(y_true,y_pred))
    print("V-measure score (n_components = %i): " %r, v_measure_score(y_true,y_pred))
    print("Adjusted Rand score (n_components = %i): " %r, adjusted_rand_score(y_true,y_pred))
    print("Adjusted mutual information score: (n_components = %i): " %r, adjusted_mutual_info_score(y_true,y_pred), "\n")
    h_scores.append(homogeneity_score(y_true,y_pred))
    c_scores.append(completeness_score(y_true,y_pred))
    vm_scores.append(v_measure_score(y_true,y_pred))
    ar_scores.append(adjusted_rand_score(y_true,y_pred))
    ami_scores.append(adjusted_mutual_info_score(y_true,y_pred))
    
#Set up histograms
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
    
plot_histogram('NMF Kullback-Leibler Homogeneity Scores', h_scores)
plot_histogram('NMF Kullback-Leibler Completeness Scores', c_scores)
plot_histogram('NMF Kullback-Leibler V-Measure Scores', vm_scores)
plot_histogram('NMF Kullback-Leibler Adjusted Rand Index Scores', ar_scores)
plot_histogram('NMF Kullback-Leibler Adjusted Mutual Information Scores', ami_scores)

### NMF Kullback-Leibler with 5 n_components is the BEST PERFORMER by far

########################### UMAP ###########################

def umap_dimension_reduction(training_data, n_components=5, metric='cosine', disconnection_distance=None, random_state=0):
    start = time.time()
    umap = UMAP(n_components=n_components, metric=metric, disconnection_distance=disconnection_distance, random_state=random_state)
    umap_dataset = umap.fit_transform(training_data)
    print("UMAP complete after %f seconds" % (time.time()-start))
    return umap_dataset, umap

#Euclidean UMAP
umap_euc_hscores = []
umap_euc_cscores = []
umap_euc_vmscores = []
umap_euc_arscores = []
umap_euc_amiscores = []

for r in [1, 2, 3, 5, 10, 20, 50, 100, 300]:
    print('n_components:',r)
    umap_dataset, umap = umap_dimension_reduction(X_train_tfidf, n_components = r, metric='euclidean', random_state=0)
    km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
    cat_index_a=np.array(cat_index)
    y_true = cat_index_a
    y_pred = km.fit_predict(umap_dataset)
    con_mat = contingency_matrix(y_true,y_pred)
    pp.pprint(con_mat)
    
    cm = confusion_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(cm, maximize=True)
    new_cm = cm[rows[:,np.newaxis], cols]
    plot_contingency_table(new_cm, title= 'n_components = %i' %r)
    print("Homogeneity score (n_components = %i): " %r, homogeneity_score(y_true,y_pred))
    print("Completeness score (n_components = %i): " %r, completeness_score(y_true,y_pred))
    print("V-measure score (n_components = %i): " %r, v_measure_score(y_true,y_pred))
    print("Adjusted Rand score (n_components = %i): " %r, adjusted_rand_score(y_true,y_pred))
    print("Adjusted mutual information score: (n_components = %i): " %r, adjusted_mutual_info_score(y_true,y_pred), "\n")
    umap_euc_hscores.append(homogeneity_score(y_true,y_pred))
    umap_euc_cscores.append(completeness_score(y_true,y_pred))
    umap_euc_vmscores.append(v_measure_score(y_true,y_pred))
    umap_euc_arscores.append(adjusted_rand_score(y_true,y_pred))
    umap_euc_amiscores.append(adjusted_mutual_info_score(y_true,y_pred))
    
plot_histogram('UMAP Euclidean Homogeneity Scores', umap_euc_hscores)
plot_histogram('UMAP Euclidean Completeness Scores', umap_euc_cscores)
plot_histogram('UMAP Euclidean V-Measure Scores', umap_euc_vmscores)
plot_histogram('UMAP Euclidean Adjusted Rand Index Scores', umap_euc_arscores)
plot_histogram('UMAP Euclidean Adjusted Mutual Information Scores', umap_euc_amiscores)

#cosine UMAP
umap_cos_hscores = []
umap_cos_cscores = []
umap_cos_vmscores = []
umap_cos_arscores = []
umap_cos_amiscores = []

for r in [1, 2, 3, 5, 10, 20, 50, 100, 300]:
    print('n_components:',r)
    umap_dataset, umap = umap_dimension_reduction(X_train_tfidf, n_components = r, metric='cosine', disconnection_distance=2, random_state=0)
    km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=40)
    cat_index_a=np.array(cat_index)
    y_true = cat_index_a
    y_pred = km.fit_predict(umap_dataset)
    con_mat = contingency_matrix(y_true,y_pred)
    pp.pprint(con_mat)
    
    cm = confusion_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(cm, maximize=True)
    new_cm = cm[rows[:,np.newaxis], cols]
    plot_contingency_table(new_cm, title= 'n_components = %i' %r)
    print("Homogeneity score (n_components = %i): " %r, homogeneity_score(y_true,y_pred))
    print("Completeness score (n_components = %i): " %r, completeness_score(y_true,y_pred))
    print("V-measure score (n_components = %i): " %r, v_measure_score(y_true,y_pred))
    print("Adjusted Rand score (n_components = %i): " %r, adjusted_rand_score(y_true,y_pred))
    print("Adjusted mutual information score: (n_components = %i): " %r, adjusted_mutual_info_score(y_true,y_pred), "\n")
    umap_cos_hscores.append(homogeneity_score(y_true,y_pred))
    umap_cos_cscores.append(completeness_score(y_true,y_pred))
    umap_cos_vmscores.append(v_measure_score(y_true,y_pred))
    umap_cos_arscores.append(adjusted_rand_score(y_true,y_pred))
    umap_cos_amiscores.append(adjusted_mutual_info_score(y_true,y_pred))
    
plot_histogram('UMAP Cosine Homogeneity Scores', umap_cos_hscores)
plot_histogram('UMAP Cosine Completeness Scores', umap_cos_cscores)
plot_histogram('UMAP Cosine V-Measure Scores', umap_cos_vmscores)
plot_histogram('UMAP Cosine Adjusted Rand Index Scores', umap_cos_arscores)
plot_histogram('UMAP Cosine Adjusted Mutual Information Scores', umap_cos_amiscores)

#Best n_components for both Euclidean and Cosine UMAP = 2

########################### AgglomerativeClustering ###########################
umap_dataset, umap = umap_dimension_reduction(X_train_tfidf, n_components = 2, metric='cosine', disconnection_distance=2, random_state=0)


for linkage in ('ward', 'single'):
    print('\n','-'*20, "Linkage:", linkage, '-'*20)

    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=5)
   
    cat_index_a=np.array(cat_index)
    y_true = cat_index_a
    y_pred = clustering.fit_predict(umap_dataset)
    con_mat = contingency_matrix(y_true,y_pred)
    pp.pprint(con_mat)
    
    cm = confusion_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(cm, maximize=True)
    new_cm = cm[rows[:,np.newaxis], cols]
    plot_contingency_table(new_cm, title= 'Linkage Criteria = %s' %linkage)
    
    print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
    print("Completeness score: ",completeness_score(y_true,y_pred))
    print("V-measure score: ",v_measure_score(y_true,y_pred))
    print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
    print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))
    
#Ward linkage criteria significantly better

########################### DBSCAN ###########################

#cosine, euclidean, l1 vs eps
umap_homos = []
umap_comp = []
umap_v_measure = []
umap_adj_rand = []
umap_adj_mutual_info = []

epsVals = [x*0.1 for x in range(3, 8)]
metrics = ["cosine", "euclidean", "l1"]

for metric_ in metrics:
    for epsVal in epsVals:
        clustering = DBSCAN(min_samples=100, eps=epsVal, metric=metric_)
        clustering_data = clustering.fit(umap_dataset)
        
        umap_homos.append(homogeneity_score(cat_index_a, clustering_data.labels_))
        umap_comp.append(completeness_score(cat_index_a, clustering_data.labels_))
        umap_v_measure.append(v_measure_score(cat_index_a, clustering_data.labels_))
        umap_adj_rand.append(adjusted_rand_score(cat_index_a, clustering_data.labels_))
        umap_adj_mutual_info.append(adjusted_mutual_info_score(cat_index_a, clustering_data.labels_))

numEpsVals = len(epsVals)
for index, metric_ in enumerate(metrics):
    plt.title('UMAP + DBSCAN using ' + metric_ + " v.s. eps") 
    plt.plot(epsVals, umap_homos[index*numEpsVals:(index+1)*numEpsVals], label='Homogeneity score')
    plt.plot(epsVals, umap_comp[index*numEpsVals:(index+1)*numEpsVals], label='Completness score')
    plt.plot(epsVals, umap_v_measure[index*numEpsVals:(index+1)*numEpsVals], label='V measure score')
    plt.plot(epsVals, umap_adj_rand[index*numEpsVals:(index+1)*numEpsVals], label='Adjusted Random score')
    plt.plot(epsVals, umap_adj_mutual_info[index*numEpsVals:(index+1)*numEpsVals], label='Adjusted Mutual Info score')
    plt.legend(loc='best')
    plt.show()
    
#cosine, euclidean, l1 vs. min_samples

umap_homos = []
umap_comp = []
umap_v_measure = []
umap_adj_rand = []
umap_adj_mutual_info = []

metrics = ["cosine", "euclidean", "l1"]
min_sampleVals = list(range(1, 30, 3))
optimalEpsVal = {"cosine":0.5, "euclidean":0.3, "l1":0.5}

for metric_ in metrics:
    for min_samplesVal in min_sampleVals:
        epsVal = optimalEpsVal[metric_]
        clustering = DBSCAN(min_samples=min_samplesVal, eps=epsVal, metric=metric_)
        clustering_data = clustering.fit(umap_dataset)
        
        umap_homos.append(homogeneity_score(cat_index_a, clustering_data.labels_))
        umap_comp.append(completeness_score(cat_index_a, clustering_data.labels_))
        umap_v_measure.append(v_measure_score(cat_index_a, clustering_data.labels_))
        umap_adj_rand.append(adjusted_rand_score(cat_index_a, clustering_data.labels_))
        umap_adj_mutual_info.append(adjusted_mutual_info_score(cat_index_a, clustering_data.labels_))

interval = len(min_sampleVals)
for index, metric_ in enumerate(metrics):
    plt.title('UMAP + DBSCAN using ' + metric_ + " v.s. min_samples") 
    plt.plot(min_sampleVals, umap_homos[index*interval:(index+1)*interval], label='Homogeneity score')
    plt.plot(min_sampleVals, umap_comp[index*interval:(index+1)*interval], label='Completness score')
    plt.plot(min_sampleVals, umap_v_measure[index*interval:(index+1)*interval], label='V measure score')
    plt.plot(min_sampleVals, umap_adj_rand[index*interval:(index+1)*interval], label='Adjusted Random score')
    plt.plot(min_sampleVals, umap_adj_mutual_info[index*interval:(index+1)*interval], label='Adjusted Mutual Info score')
    plt.legend(loc='best')
    plt.show()
    
    

umap_homos = {}
umap_comp = {}
umap_v_measure = {}
umap_adj_rand = {}
umap_adj_mutual_info = {}

epsVals = [x*0.1 for x in range(3, 8)]
# pVals = [None, 1, 0.25] all the same
metrics = ["cosine", "euclidean", "l1"]
min_sampleVals = list(range(1, 100, 5))

for metric_ in metrics:
    for epsVal in epsVals:
        for min_samplesVal in min_sampleVals:
            clustering = DBSCAN(min_samples=min_samplesVal, eps=epsVal, metric=metric_)
            clustering_data = clustering.fit(umap_dataset)
  
            umap_homos[(metric_, epsVal, min_samplesVal)] = homogeneity_score(cat_index_a, clustering_data.labels_)
            umap_comp[(metric_, epsVal, min_samplesVal)] = completeness_score(cat_index_a, clustering_data.labels_)
            umap_v_measure[(metric_, epsVal, min_samplesVal)] = v_measure_score(cat_index_a, clustering_data.labels_)
            umap_adj_rand[(metric_, epsVal, min_samplesVal)] = adjusted_rand_score(cat_index_a, clustering_data.labels_)
            umap_adj_mutual_info[(metric_, epsVal, min_samplesVal)] = adjusted_mutual_info_score(cat_index_a, clustering_data.labels_)

sorted(list(umap_v_measure.items()),key=lambda x: x[1], reverse=True)

umap_dataset, umap = umap_dimension_reduction(X_train_tfidf, n_components = 2, metric='euclidean', disconnection_distance=2, random_state=0)

clustering = DBSCAN(min_samples=46, eps=0.7, metric='euclidean')

newCm = []
for index, i in enumerate(cm):
    if i.sum() != 0:
        newCm.append(i)
newCm = np.array(newCm)

cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = clustering.fit_predict(umap_dataset)
con_mat = contingency_matrix(y_true,y_pred)
pp.pprint(con_mat)

#cm = newCm
#rows, cols = linear_sum_assignment(cm, maximize=True)
#new_cm = cm[rows[:,np.newaxis], cols]
#plot_contingency_table(new_cm, title= 'DBSCAN Optimized')
    
print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))

########################### HDBSCAN ###########################

umap_homos = []
umap_comp = []
umap_v_measure = []
umap_adj_rand = []
umap_adj_mutual_info = []

# pVals = [None, 1, 0.25] all the same
metrics = ["euclidean", "l1"]
min_sampleVals = [None] + list(range(1, 20, 2))

for metric_ in metrics:
    for min_samplesVal in min_sampleVals:
        clustering = hdbscan.HDBSCAN(metric=metric_, min_samples=min_samplesVal, min_cluster_size=100)
        clustering_data = clustering.fit(umap_dataset)
        
        umap_homos.append(homogeneity_score(cat_index_a, clustering_data.labels_))
        umap_comp.append(completeness_score(cat_index_a, clustering_data.labels_))
        umap_v_measure.append(v_measure_score(cat_index_a, clustering_data.labels_))
        umap_adj_rand.append(adjusted_rand_score(cat_index_a, clustering_data.labels_))
        umap_adj_mutual_info.append(adjusted_mutual_info_score(cat_index_a, clustering_data.labels_))

interval = len(min_sampleVals)
modMin_sampleVals = [0] + min_sampleVals[1:]
for index, metric_ in enumerate(metrics):
    plt.title('UMAP + HDBSCAN using ' + metric_ + " v.s. min_samples") 
    plt.plot(modMin_sampleVals, umap_homos[index*interval:(index+1)*interval], label='Homogeneity score')
    plt.plot(modMin_sampleVals, umap_comp[index*interval:(index+1)*interval], label='Completness score')
    plt.plot(modMin_sampleVals, umap_v_measure[index*interval:(index+1)*interval], label='V measure score')
    plt.plot(modMin_sampleVals, umap_adj_rand[index*interval:(index+1)*interval], label='Adjusted Random score')
    plt.plot(modMin_sampleVals, umap_adj_mutual_info[index*interval:(index+1)*interval], label='Adjusted Mutual Info score')
    plt.legend(loc='best')
    plt.show()
    
#HDBSCAN Contingency Matrix and Scores

clustering = hdbscan.HDBSCAN(min_cluster_size=100, metric="euclidean", min_samples=20)
clustering_data = clustering.fit(umap_dataset)

cat_index_a=np.array(cat_index)
y_true = cat_index_a
y_pred = clustering.fit_predict(umap_dataset)
print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))

cm = confusion_matrix(cat_index_a, clustering_data.labels_)
rows, cols = linear_sum_assignment(cm, maximize=True) 
x=cm[rows[:, np.newaxis], cols]
plot_contingency_table(x)    

            