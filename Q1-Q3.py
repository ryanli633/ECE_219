from sklearn.datasets import fetch_20newsgroups
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
#Custom stop words
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from string import punctuation
import string
from nltk import pos_tag
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.utils.extmath import randomized_svd

#pulling data from 20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train',
                                     shuffle=True,
                                     random_state=42)

categories=list(newsgroups_train.target_names)

###################### QUESTION 1 ######################
print('='*20, 'QUESTION 1', '='*20)

targets=newsgroups_train.target.tolist()  #list of all document tagets
cat_count=[]
for n in range(20):    #looping through unique targets (0-19) and getting count
    count = targets.count(n)
    cat_count.append(count)

#Creating bar graph    
plt.figure(figsize=(10, 6))
plt.title('Number of documents for each category', fontsize=20)
plt.xticks(rotation='vertical', fontsize=12)
plt.xlabel('Category Targets', fontsize=16)
plt.ylabel('Number of docs', fontsize=16)
plt.axis([-1,20,0,700])
plt.bar(categories, cat_count)

#annotateing bar graph
for x,y in zip(categories,cat_count):
    label = "{:.0f}".format(y)
    plt.annotate(label,
                (x,y),
                textcoords="offset points",
                xytext=(0,14),
                ha='center',
                fontsize=10)
    
###################### QUESTION 2 ######################
print('='*20, 'QUESTION 2', '='*20)

#Fetching subsets of interest
q2cats=['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'rec.autos', 'rec.motorcycles',
'rec.sport.baseball', 'rec.sport.hockey']

train_dataset = fetch_20newsgroups(subset = 'train',
                                   categories = q2cats,
                                   shuffle = True,
                                   random_state = None)
test_dataset = fetch_20newsgroups(subset = 'test',
                                 categories = q2cats,
                                 shuffle = True,
                                 random_state = None)

#custom stop words
stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(stop_words_skt))

#Build CountVectorizer
vectorize = CountVectorizer(min_df=3, stop_words=combined_stopwords)

#check if term is a number
def number_term(t):
    try:
        float(t)
        return True
    except ValueError:
        return False
    
#morphing penn treebank tags to WordNet
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
        if(not number_term(word)):  #removing number terms
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

#vectorize lemmetized documents
lemvectrain=vectorize.fit_transform(lemmdata(train_dataset.data))
lemvectest=vectorize.transform(lemmdata(test_dataset.data))

#applying the tf-idf transformer to both datasets
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(lemvectrain)
test_tfidf = tfidf_transformer.transform(lemvectest)
print('Shape of TF-IDF train martix: ', train_tfidf.shape)
print('Shape of TF-IDF test martix: ', test_tfidf.shape)

###################### QUESTION 3 ######################
print('='*20, 'QUESTION 3', '='*20)

#LSI
svd = TruncatedSVD(n_components=50, random_state=42)
lsi_train = svd.fit_transform(train_tfidf)
lsi_test = svd.transform(test_tfidf)
U,S,Vt = randomized_svd(train_tfidf, n_components=50, random_state = 42) #left/right singular matrices & singular values
SIG = np.diag(S)
lsi_err = np.sum(np.array(train_tfidf - U.dot(SIG).dot(Vt))**2)  #||X-U_k*SIG_k*V^T_k||^2_F


#NMF
nmf = NMF(n_components = 50, init = 'random', random_state = 42)
nmf_train = nmf.fit_transform(train_tfidf)
nmf_test = nmf.transform(test_tfidf)
H = nmf.components_
nmf_err = np.sum(np.array(train_tfidf - nmf_train.dot(H))**2)  #||X-WH||^2_F

print('LSI error =', lsi_err)
print('NMF error =', nmf_err)