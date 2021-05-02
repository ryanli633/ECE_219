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
# Optimum setup: combined_stopwords, min_df=4, max_df=600

stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(stop_words_skt))

tfidf_vect = TfidfVectorizer(stop_words=combined_stopwords, min_df=4, max_df=600)
X_train_tfidf = tfidf_vect.fit_transform(train_docs)
X_test_tfidf = tfidf_vect.transform(test_docs)


########################### Clustering & Performance Evaluation ###########################
# Optimum setup: NMF with Kullback-Leibler & r=5

def nmf_dimension_reduction(training_data, n_components=None, solver='cd', beta_loss='frobenius', max_iter=1000, random_state=0):
    start = time.time()
    nmf = NMF(n_components=n_components, solver=solver, init='nndsvda', beta_loss=beta_loss, max_iter=max_iter, random_state=random_state)
    nmf_dataset = nmf.fit_transform(training_data)
    print("NMF complete after %f seconds" % (time.time()-start))
    return nmf_dataset, nmf

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


#creating a digit form of the five labels
cat_index=[]
for i in bbc_cats:
    if i == 'business': cat_index.append(0)
    if i == 'entertainment': cat_index.append(1)
    if i == 'politics': cat_index.append(2)
    if i == 'sport': cat_index.append(3)
    if i == 'tech': cat_index.append(4)


r=5
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

########################### Applying to test docs ###########################

nmf_dataset_t, nmf = nmf_dimension_reduction(X_test_tfidf, n_components = 5, solver='mu', beta_loss='kullback-leibler', max_iter=1000, random_state=0)

r_best_nmf = 5

km = KMeans(n_clusters=5, random_state=0, max_iter=1000, n_init=90)

nmf_labels = km.fit_predict(nmf_dataset_t)
plt.figure(figsize=(10, 6))
plt.scatter(nmf_dataset_t[:,0],nmf_dataset_t[:,1],c=nmf_labels,cmap='viridis')
plt.title("NMF with KL Clustering labels (r= %i)" %r_best_nmf);