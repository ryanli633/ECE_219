#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[7]:


import sys
import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
import pandas as pd

import random
random.seed(42)

import nltk
from nltk import pos_tag
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# In[8]:


# Question 4:

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
import itertools
from sklearn.model_selection import GridSearchCV

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def print_classifier_metrics(y_test,y_pred,name="",average='binary'):
    print("Accuracy score for %s: %f" %(name,accuracy_score(y_test,y_pred)))
    print("Recall score for %s: %f" % (name,recall_score(y_test,y_pred,average=average)))
    print("Precision score for %s: %f" % (name,precision_score(y_test,y_pred,average=average)))
    print("F-1 score for %s: %f" % (name,f1_score(y_test,y_pred,average=average)))

def plot_roc_curve(y_test,decision_function,name=""):
    fpr = dict();tpr = dict();roc_auc = dict()
    fpr, tpr, thresholds = roc_curve(y_test, decision_function)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');
    plt.title('%s ROC curve' % name);plt.legend(loc="lower right")

hardSVM = LinearSVC(C=1000,random_state=42)
softSVM = LinearSVC(C=0.0001,random_state=42)

y_train = [int(i/4) for i in train_dataset.target] 
y_test = [int(i/4) for i in test_dataset.target] 

y_pred_hardSVM = hardSVM.fit(X_train_LSI,y_train).predict(X_test_LSI) 
y_pred_softSVM = softSVM.fit(X_train_LSI,y_train).predict(X_test_LSI) 

print_classifier_metrics(y_test,y_pred_hardSVM,name="Hard Margin SVM")
print_classifier_metrics(y_test,y_pred_softSVM,name="Soft Margin SVM")

class_names = ['Computer Technology', 'Recreation Activity']
hardSVM_cm = confusion_matrix(y_test,y_pred_hardSVM) 
plt.figure(); plot_confusion_matrix(hardSVM_cm, classes=class_names, title='Hard SVM Confusion Matrix') 
softSVM_cm = confusion_matrix(y_test,y_pred_softSVM) 
plt.figure(); plot_confusion_matrix(softSVM_cm, classes=class_names, title='Soft SVM Confusion Matrix')

plot_roc_curve(y_test,hardSVM.decision_function(X_test_LSI),name="Hard Margin SVM")  
plot_roc_curve(y_test,softSVM.decision_function(X_test_LSI),name="Soft Margin SVM")  

svc = LinearSVC(random_state=42) 
params = {'C':[0.001,0.01,0.1,1,10,100,1000]}
clf = GridSearchCV(svc,params,cv=5,scoring='accuracy') 
clf.fit(X_train_LSI,y_train)

x = [0.001,0.01,0.1,1,10,100,1000]
y = clf.cv_results_['mean_test_score']
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(x,y)
for xy in zip(x, y):                                       
    ax.annotate('(%s, %.5f)' % xy, xy=xy, textcoords='data')
plt.xlabel('C'); plt.ylabel('Mean Test Score'); plt.title("Linear SVM")
plt.grid()

y_pred_cv = clf.best_estimator_.predict(X_test_LSI)
best_svm_gamma = clf.best_estimator_.C

print("Grid search results for SVM: ", clf.cv_results_)
print("Best estimator for SVM: ", clf.best_estimator_)
print("Best parameters for SVM: ", clf.best_params_)
print("Best score for SVM: ", clf.best_score_)
print("Best Gamma for SVM: ", best_svm_gamma)
print_classifier_metrics(y_test,y_pred_cv,name="Best Gamma SVM")

cv_cm = confusion_matrix(y_test,y_pred_cv) 
plt.figure(); plot_confusion_matrix(cv_cm, classes=class_names, title='Best Gamma SVM Confusion Matrix')

plot_roc_curve(y_test,clf.best_estimator_.decision_function(X_test_LSI),name="Best Gamma SVM")  


# In[9]:


# Question 5:

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=10**10,random_state=42) 
y_pred_lr = lr.fit(X_train_LSI,y_train).predict(X_test_LSI)
print("Coefficients learned by logistic regression without regularization: ", lr.coef_)
print_classifier_metrics(y_test,y_pred_lr,name="Logistic Regression without regularization")
lr_cm = confusion_matrix(y_test,y_pred_lr) 
plt.figure(); plot_confusion_matrix(lr_cm, classes=class_names, title='Logistic Regression Confusion Matrix')
plot_roc_curve(y_test,lr.decision_function(X_test_LSI),name="Logistic Regression") 
lr_l2 = LogisticRegression(random_state=42) 
clf_l2 = GridSearchCV(lr_l2,params,cv=5,scoring='accuracy') 
y_pred_l2 = clf_l2.fit(X_train_LSI,y_train).best_estimator_.predict(X_test_LSI)
best_l2_gamma = clf_l2.best_estimator_.C

x = [0.001,0.01,0.1,1,10,100,1000]
y = clf_l2.cv_results_['mean_test_score']
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(x,y)
for xy in zip(x, y):                                       
    ax.annotate('(%s, %.5f)' % xy, xy=xy, textcoords='data')
plt.xlabel('C'); plt.ylabel('Mean Test Score'); plt.title("Logistic Regression with L2 regularization")
plt.grid()


print("Grid search results for logistic regression with l-2 regularization: ", clf_l2.cv_results_)
print("Best estimator for logistic regression with l-2 regularization: ", clf_l2.best_estimator_)
print("Best parameters for logistic regression with l-2 regularization: ", clf_l2.best_params_)
print("Best score for logistic regression with l-2 regularization: ", clf_l2.best_score_)
print("Best Gamma for logistic regression with l-2 regularization: ", best_l2_gamma)
print("Coefficients learned by logistic regression with l-2 regularization: ", clf_l2.best_estimator_.coef_)
print_classifier_metrics(y_test,y_pred_l2,name="Logistic Regression with l-2 regularization")


lr_l1 = LogisticRegression(penalty='l1',random_state=42) 
clf_l1 = GridSearchCV(lr_l1,params,cv=5,scoring='accuracy') 
y_pred_l1 = clf_l1.fit(X_train_LSI,y_train).best_estimator_.predict(X_test_LSI)
best_l1_gamma = clf_l1.best_estimator_.C

x = [0.001,0.01,0.1,1,10,100,1000]
y = clf_l1.cv_results_['mean_test_score']
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(x,y)
for xy in zip(x, y):                                       
    ax.annotate('(%s, %.5f)' % xy, xy=xy, textcoords='data')
plt.xlabel('C'); plt.ylabel('Mean Test Score'); plt.title("Logistic Regression with L1 regularization")
plt.grid()

print("Grid search results for logistic regression with l-1 regularization: ", clf_l1.cv_results_)
print("Best estimator for logistic regression with l-1 regularization: ", clf_l1.best_estimator_)
print("Best parameters for logistic regression with l-1 regularization: ", clf_l1.best_params_)
print("Best score for logistic regression with l-1 regularization: ", clf_l1.best_score_)
print("Best Gamma for logistic regression with l-1 regularization: ", best_l1_gamma)
print("Coefficients learned by logistic regression with l-1 regularization: ", clf_l1.best_estimator_.coef_)
print_classifier_metrics(y_test,y_pred_l1,name="Logistic Regression with l-1 regularization")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




