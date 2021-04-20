import sys
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import string
from string import punctuation

import itertools
import re


# # Data Preprocessing

from sklearn.datasets import fetch_20newsgroups
categories=['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

# Training dataset with no removal of headers and footers and no lemmatization
X_train = fetch_20newsgroups(subset = 'train',
                                   categories = categories,
                                   shuffle = True,
                                   random_state = 42)
# Test dataset with no removal of headers and footers and no lemmatization
X_test = fetch_20newsgroups(subset = 'test',
                                 categories = categories,
                                 shuffle = True,
                                 random_state = 42)
# Training dataset WITH REMOVAL OF HEADERS AND FOOTERS and no lemmatization
X_train_r = fetch_20newsgroups(subset = 'train', 
                                           categories = categories, 
                                           remove = ('headers', 'footers'),
                                           shuffle = True, 
                                           random_state = 42)
# Test dataset WITH REMOVAL OF HEADERS AND FOOTERS and no lemmatization
X_test_r = fetch_20newsgroups(subset = 'test',
                                 categories = categories,
                                 remove = ('headers', 'footers'),
                                 shuffle = True,
                                 random_state = 42)


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(lemmatizer, sentence):
    return  ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])

def preprocessing(lemmatizer, dataset):
    corpus = [re.sub(r"[0123456789.-]", " ", sentence) for sentence in dataset]
    lemmatizedCorpus = [lemmatize(lemmatizer, sentence) for sentence in corpus]
    return lemmatizedCorpus

lemmatizer = WordNetLemmatizer()
# Training dataset with no removal of headers and footers and LEMMATIZATION
X_train_l = preprocessing(lemmatizer, X_train.data)
# Test dataset with no removal of headers and footers and LEMMATIZATION
X_test_l = preprocessing(lemmatizer, X_test.data)

lemmatizer_r = WordNetLemmatizer()
# Training dataset WITH REMOVAL OF HEADERS AND FOOTERS AND LEMMATIZATION
X_train_rl = preprocessing(lemmatizer_r, X_train_r.data)
# Test dataset WITH REMOVAL OF HEADERS AND FOOTERS AND LEMMATIZATION
X_test_rl = preprocessing(lemmatizer_r, X_test_r.data)


# Our final targets are in 2 categories: "Computer Technology" and "Recreational Activity"
# Convert 8 imported categories into 2 categories
y_train = [int(i/4) for i in X_train.target] 
y_test = [int(i/4) for i in X_test.target] 


# stop words
stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))


# # Pipeline and GridSearchCV

# used to cache results
from tempfile import mkdtemp
from shutil import rmtree
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import Memory
cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=10)

pipeline = Pipeline([
    ('feat_extract', CountVectorizer(min_df=3, stop_words=combined_stopwords)),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(random_state=42)),
    ('clf', GaussianNB()),
],
memory=memory
)

MIN_DF_OPTIONS = [3, 5]

param_grid = [
    {
        'feat_extract': [CountVectorizer(min_df=3, stop_words=combined_stopwords)], # 1 choice
        'feat_extract__min_df': MIN_DF_OPTIONS, # 2 choices
        'reduce_dim': [TruncatedSVD(n_components=50,random_state=42), NMF(n_components=50,random_state=42)], # 2 choices
        'clf': [LinearSVC(C=100,max_iter=100000,random_state=42), 
                LogisticRegression(penalty='l1',C=10,solver='saga',max_iter=1000,random_state=42),
                LogisticRegression(penalty='l2',C=100,max_iter=1000,random_state=42),
                GaussianNB()], # 4 choices
    },
]

grid = GridSearchCV(pipeline, cv=5, n_jobs=-1, param_grid=param_grid, scoring='accuracy')
grid.fit(X_train.data, y_train)

grid_r = GridSearchCV(pipeline, cv=5, n_jobs=-1, param_grid=param_grid, scoring='accuracy')
grid_r.fit(X_train_r.data, y_train)

grid_l = GridSearchCV(pipeline, cv=5, n_jobs=-1, param_grid=param_grid, scoring='accuracy')
grid_l.fit(X_train_l, y_train)

grid_rl = GridSearchCV(pipeline, cv=5, n_jobs=-1, param_grid=param_grid, scoring='accuracy')
grid_rl.fit(X_train_rl, y_train)
rmtree(cachedir)


# Results with no removal of headers or footers and no lemmatization
print("Best parameters (no r, no l): ", grid.best_params_)
print("Best score on training data (no r, no l): ", grid.best_score_)
print("Accuracy on test data (no r, no l): ", grid.score(X_test.data, y_test))
display(pd.DataFrame(grid.cv_results_))

# Results WITH REMOVAL OF HEADERS AND FOOTERS and no lemmatization
print("Best parameters (r, no l): ", grid_r.best_params_)
print("Best score on training data (r, no l): ", grid_r.best_score_)
print("Accuracy on test data (r, no l): ", grid_r.score(X_test_r.data, y_test))
display(pd.DataFrame(grid_r.cv_results_))

# Results with no removal of headers and footers and LEMMATIZATION
print("Best parameters (no r, l): ", grid_l.best_params_)
print("Best score on training data (no r, l): ", grid_l.best_score_)
print("Accuracy on test data (no r, l): ", grid_l.score(X_test_l, y_test))
display(pd.DataFrame(grid_l.cv_results_))

# Results WITH REMOVAL OF HEADERS AND FOOTERS AND LEMMATIZATION
print("Best parameters (r, l): ", grid_rl.best_params_)
print("Best score on training data (r, l): ", grid_rl.best_score_)
print("Accuracy on test data (r, l): ", grid_rl.score(X_test_rl, y_test))
display(pd.DataFrame(grid_rl.cv_results_))

# Best results compared
print("Best score on training data (no r, no l): ", grid.best_score_)
print("Test score (no r, no l): ", grid.score(X_test.data, y_test))
print("Best score on training data (r, no l): ", grid_r.best_score_)
print("Test score (r, no l): ", grid_r.score(X_test_r.data, y_test))
print("Best score on training data (no r, l): ", grid_l.best_score_)
print("Test score (no r, l): ", grid_l.score(X_test_l, y_test))
print("Best score on training data (r, l): ", grid_rl.best_score_)
print("Test score (r, l): ", grid_rl.score(X_test_rl, y_test))

# Parameters of best results
print("Best parameters (no r, no l): ", grid.best_params_)
print("Best parameters (r, no l): ", grid_r.best_params_)
print("Best parameters (no r, l): ", grid_l.best_params_)
print("Best parameters (r, l): ", grid_rl.best_params_)