from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn import svm

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

###################### QUESTION 12 ######################
print('='*20, 'QUESTION 12', '='*20)

#categories of interest
q12cat = [
'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware',
'misc.forsale', 
'soc.religion.christian']

#matching class names
q12class = [
    'IBM HW',
    'MAC HW',
    'Forsale',
    'Christianity']

#fetching data
q12trainds = fetch_20newsgroups(subset= 'train', categories = q12cat, shuffle = True, random_state = 42)
q12testds = fetch_20newsgroups(subset= 'test', categories = q12cat, shuffle = True, random_state = 42)

#Lemmatize and vectorize
q12lemvectrain=vectorize.fit_transform(lemmdata(q12trainds.data))
q12lemvectest=vectorize.transform(lemmdata(q12testds.data))

#tf_idf transformation
tfidf_transformer = TfidfTransformer()
q12trn_tfidf = tfidf_transformer.fit_transform(q12lemvectrain)
q12tst_tfidf = tfidf_transformer.transform(q12lemvectest)

#LSI
svd = TruncatedSVD(n_components=50, random_state=42)
lsi_train = svd.fit_transform(q12trn_tfidf)
lsi_test = svd.transform(q12tst_tfidf)

#Pulling Targets
q12tr_targ = q12trainds.target
q12ts_targ = q12testds.target

#defining required classifier metrics
def print_classifier_metrics(y_test,y_pred,name="",average='binary'):
    print("Accuracy score for %s: %f" %(name,accuracy_score(y_test,y_pred)))
    print("Recall score for %s: %f" % (name,recall_score(y_test,y_pred,average=average)))
    print("Precision score for %s: %f" % (name,precision_score(y_test,y_pred,average=average)))
    print("F-1 score for %s: %f" % (name,f1_score(y_test,y_pred,average=average)))

#defining Confusion Matrix plot
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 5))
    plt.xlabel('Predicted label') 
    plt.ylabel('True label')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")


################# Gaussian Naive Bayes #################
print('='*10, 'Gaussian Naive Bayes', '='*10)

#Gaussian Naive Bayes
X_train_LSI=lsi_train
X_test_LSI=lsi_test
y_train = q12tr_targ
y_test = q12ts_targ

GNB = GaussianNB()
y_pred_GNB = GNB.fit(X_train_LSI, y_train).predict(X_test_LSI)

#GNB metrics
print_classifier_metrics(y_test, y_pred_GNB, average='weighted', name="Multiclass GNB")

#GNB confusion matrix
gnb_conf_matrix = confusion_matrix(y_test,y_pred_GNB)
class_names = q12class
plt.figure(); plot_confusion_matrix(gnb_conf_matrix, classes=class_names, title="Confusion matrix, Multiclass Gaussian Naive Bayes")
plt.figure(); plot_confusion_matrix(gnb_conf_matrix, classes=class_names, normalize=True, title="Normalized Confusion matrix, Multiclass Gaussian Naive Bayes")


################# SVM One vs. One #################
print('='*10, 'SVM One vs. One', '='*10)

x = {'estimator__C':[0.001,0.01,0.1,1,10,100,1000]} #gamma parameters in Linear SVC
svm_1v1 = OneVsOneClassifier(LinearSVC(random_state = 42)) #1v1 classifier for SVM
gsg_1v1 = GridSearchCV(svm_1v1,x,cv=5,scoring='accuracy') #best gamma grid search
y_pred_1v1 = gsg_1v1.fit(X_train_LSI, y_train).best_estimator_.predict(X_test_LSI) #best estimator fit
#print(gsg_1v1.best_estimator_) #best estimator: C=10
print_classifier_metrics(y_test,y_pred_1v1,name="One vs One SVM",average='weighted')

#SVM One vs. One Confusion Matrix  
svm1v1_cm = confusion_matrix(y_test,y_pred_1v1)
plt.figure(); plot_confusion_matrix(svm1v1_cm, classes=class_names, title='Confusion matrix, Multiclass SVM 1 vs 1')
plt.figure(); plot_confusion_matrix(svm1v1_cm, classes=class_names, normalize=True, title='Normalized Confusion matrix, Multiclass SVM 1 vs 1')

################# SVM One vs. Rest #################
print('='*10, 'SVM One vs. Rest', '='*10)

svm_1vR = OneVsRestClassifier(LinearSVC(random_state = 42)) #1vRest classifier for SVM
gsg_1vR = GridSearchCV(svm_1vR,x,cv=5,scoring='accuracy') #best gamma grid search
y_pred_1vR = gsg_1vR.fit(X_train_LSI, y_train).best_estimator_.predict(X_test_LSI) #best estimator fit
#print(gsg_1vR.best_estimator_) #best estimator: C=10
print_classifier_metrics(y_test,y_pred_1vR,name="One vs Rest SVM",average='weighted')

#SVM One vs. Rest Confusion Matrix   
svm1vR_cm = confusion_matrix(y_test,y_pred_1vR)
plt.figure(); plot_confusion_matrix(svm1vR_cm, classes=class_names, title='Confusion matrix, Multiclass SVM 1 vs Rest')
plt.figure(); plot_confusion_matrix(svm1vR_cm, classes=class_names, normalize=True, title='Normalized Confusion matrix, Multiclass SVM 1 vs Rest')
