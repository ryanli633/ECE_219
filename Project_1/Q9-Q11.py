from matplotlib import pyplot as plt
import numpy as np
# %matplotlib inline
from sklearn.datasets import fetch_20newsgroups
# Refer to the offcial document of scikit-learn for detailed usages:
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
twenty_train = fetch_20newsgroups(subset='train', # choose which subset of the dataset to use; can be 'train', 'test', 'all'
                                  categories=None, # choose the categories to load; if is `None`, load all categories
                                  shuffle=True,
                                  random_state=42,  #set the seed of random number generator when shuffling to make the outcome repeatable across different runs
                                  remove=['headers', 'footers']
                                 )

import random
import sklearn
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
from sklearn.datasets import fetch_20newsgroups
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet

np.random.seed(42)
random.seed(42)

categories = ['comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']

train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = None)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = None)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(lemmatizer, sentence):
  #word_list = nltk.word_tokenize(line)
  #lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
  return  ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])

def preprocessing(lemmatizer, dataset):
  corpus = [re.sub(r"[0123456789.-]", " ", sentence) for sentence in dataset]
  lemmatizedCorpus = [lemmatize(lemmatizer, sentence) for sentence in corpus]
  return lemmatizedCorpus

lemmatizer = WordNetLemmatizer()
preprocessedTraining = preprocessing(lemmatizer, train_dataset.data)
preprocessedTesting = preprocessing(lemmatizer, test_dataset.data)

vectorizer = TfidfVectorizer(stop_words='english', min_df=3)
X_train = vectorizer.fit_transform(preprocessedTraining)
X_test = vectorizer.transform(preprocessedTesting)

import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics

!wget "http://nlp.stanford.edu/data/glove.6B.zip"
!unzip glove.6B.zip

dimension_of_glove = 300
  
glovefile = "glove.6B." + str(dimension_of_glove) + "d.txt"
embeddings_dict = {}

with open(glovefile, 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

removeKeys = ["From", "Reply-To", "Lines", "Originator", 'Nntp-Posting-Host', 'Article-I.D.']

def removeHeader(dataset):
  output = []
  for i in dataset:
    curr = ""
    headers = i[:i.find("\n\n")].split("\n")
    for row in headers:
      items = row.split(":")
      if items[0] not in removeKeys:
        curr += " ".join(items[1:])
    curr += i[i.find("\n\n"):]
    output.append(curr)
  return output


def gloveEmbedding(documents):
  output = np.empty((0,dimension_of_glove))
  for x in documents:
    gloves = np.array([embeddings_dict[word] for word in x if word in embeddings_dict])
    output = np.vstack((output, np.average(gloves, axis=0)))
  return output

from sklearn.preprocessing import Normalizer

preprocessedTraining1 = removeHeader(preprocessedTraining)
print(preprocessedTesting1)
preprocessedTesting1 = removeHeader(preprocessedTesting)
print(preprocessedTesting1)
X_train = gloveEmbedding(preprocessedTraining1)
X_test = gloveEmbedding(preprocessedTesting1)

Y_train = train_dataset.target
Y_test = test_dataset.target

normalizer = Normalizer()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

# soft margin SVM classifier
clf_svm2 = SVC(C=1.0, kernel='rbf')
clf_svm2.fit(X_train, Y_train.astype('int'))
score = clf_svm2.decision_function(X_test)
predicted = clf_svm2.predict(X_test)

print("%-12s %f" % ('Accuracy:', metrics.accuracy_score(Y_test.astype('int'), predicted)))
print("%-12s %f" % ('Precision:', metrics.precision_score(Y_test.astype('int'), predicted, average='macro')))
print("%-12s %f" % ('Recall:', metrics.recall_score(Y_test.astype('int'), predicted, average='macro')))
#print("%-12s %f" % ('F-1 score:', metrics.f1_score(Y_test.astype('int'), predicted)))
#print("Confusion Matrix: \n{0}".format(metrics.confusion_matrix(Y_test.astype('int'), predicted)))

!pip install scprep phate umap-learn

import umap
reducer = umap.UMAP(metric='cosine')
embedding = reducer.fit_transform( X_train)

fig, ax = plt.subplots()
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=Y_train, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
legend1 = ax.legend(*scatter.legend_elements())
ax.add_artist(legend1)
plt.title('UMAP projection of the GLoVE-based embeddings', fontsize=24)

randomVectors = normalizer.transform(np.random.randn(*X_train.shape))
randomEmbedding = reducer.transform(randomVectors)

plt.scatter(
    randomEmbedding[:, 0],
    randomEmbedding[:, 1],
    c=np.random.randint(2, size=(Y_train.shape)), cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
legend1 = ax.legend(*scatter.legend_elements())
ax.add_artist(legend1)
plt.title('UMAP projection of the normalized random vectors', fontsize=24)

randomEmbedding = reducer.fit_transform(randomVectors)
plt.scatter(
    randomEmbedding[:, 0],
    randomEmbedding[:, 1],
    c=np.random.randint(2, size=(Y_train.shape)), cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the normalized random vectors (fit_transform)', fontsize=24)


