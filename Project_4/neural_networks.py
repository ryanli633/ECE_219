from sklearn.neural_network import MLPRegressor


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import cross_validate
'''
Please read this section as it will help explain some of the parameters you need
to tune to train your neural network.

The parameter "hidden_layer_sizes", which accepts a tuple, controls the NUMBER OF 
HIDDEN NEURONS AND DEPTH
hidden_layer_sizes=(100,) represents a depth of 1 with 100 hidden neurons in that layer
hidden_layer_sizes=(100,50,) represents a depth of 2 with 100 hidden neurons in
    the first layer, and 50 hidden neurons in the second layer
At a minimum you probably want to sweep: [10, 50, 100, 150, 200] for the first layer.
You can also check other numbers of hidden neurons both within the range of [10-200]
or greater than 200. More neurons will require more computation time.

You'll want to check at least a depth of 2, although you may not need as many neurons
for each deeper level.
More depth will result in longer computation time.


The parameter "activation" controls the ACTIVATION FUNCTION that is used for the output.
activation='identity'  is what to use if you want NO ACTIVATION FUNCTION
Other values are 'relu', 'logistic', 'tanh'. You'll want to try all 3 of these, at least
with a single layer of depth to figure out which one works best for your dataset. Once
you've identified which one is best, you may not have to test all of these for subsequent
layers of depth, etc., although if you have the time, it might be good to.


The parameter "alpha" controls REGULARIZATION and can be used to adjust WEIGHT DECAY
Increasing alpha may fix overfitting. Decreasing alpha may fix underfitting. So, there is
generally a sweet spot.
The default value for alpha is alpha=0.0001
This generally is a good value for solver='adam', which is the default solver.
You can also try other powers of 10 [..., 0.01, 0.001, 0.00001, ...]
And other values when you feel like you've found a good general range for alpha:
[..., 0.0001, 0.0003, 0.0005, 0.0007, ...]


EXAMPLE 1 uses GridSearchCV, which is good as a built-in tool to automate searching
over the parameter space. But, sometimes it can be a little hard to tell if things
went wrong. If you have a lot of different parameters to search, this could take awhile.

EXAMPLE 2 uses for loops, which is a little more manual, but allows you to see some
of the details and have a better idea if something is messing up.
'''

from sklearn.neural_network import MLPRegressor

# EXAMPLE 1: Uses GridSearchCV
def cv_test(model, X, y, k=10, plot=True):
    # cross validation
    kf = KFold(n_splits=k, random_state=42)
    RMSE_train=[]
    RMSE_test=[]
    for train_index, test_index in kf.split(X):
        X_train= X[train_index,:]
        y_train= y[train_index]
        X_test= X[test_index,:]
        y_test= y[test_index]
        reg = model.fit(X_train, y_train)
        pred_train = reg.predict(X_train)
        pred_test = reg.predict(X_test)
        RMSE_train.append(np.sqrt(mean_squared_error(y_train, pred_train)))
        RMSE_test.append(np.sqrt(mean_squared_error(y_test, pred_test)))
    [rmse_train_ave, rmse_test_ave] = [np.mean(RMSE_train), np.mean(RMSE_test)]
    print('RMSE for train data=',rmse_train_ave)
    print('RMSE for test data=',rmse_test_ave)
    
    # test
    reg = model.fit(X, y)
    pred= reg.predict(X)
    if plot:
        plt.figure()
        plt.scatter(y, pred, marker='.')
        plt.plot(y,y,color='black')
        plt.xlabel('true values')
        plt.ylabel('fitted values')
        plt.title('fitted values versus true values')
        plt.show()

        plt.figure()
        plt.scatter(pred, (y - pred), marker='.')
        plt.plot(pred,np.zeros_like(pred),color='black')
        plt.xlabel('fitted values')
        plt.ylabel('residuals')
        plt.title('residuals versus fitted values')
        plt.show()
    
    return pred, rmse_train_ave, rmse_test_ave

# Assumes you have all your data, both training and test data, in (X, y)
# X = input data, y = output/target data
X = 0
y = 0

# May need to increase max_iter to get convergence
# can change to verbose=True to see progress messages
model = MLPRegressor(hidden_layer_sizes=(100,),
                     activation='relu', solver='adam', max_iter=500, 
                     alpha=1e-4, random_state=42, verbose=False)
cv_test(model, X, y, k=10)

# GridSearch to find best values
parameters = {'hidden_layer_sizes': [(10,), (50,), (100,), (150,), (200,),
                                     (100,100,), (100,200,), (200,100,)], 
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'alpha': [1e-5,1e-4,3e-4,1e-3]}
clf2 = GridSearchCV(MLPRegressor(solver='adam',max_iter=500,random_state=42,verbose=False), 
                   parameters, cv=10, scoring='neg_root_mean_squared_error', return_train_score=True)
clf2.fit(X, y)

CVres = pd.DataFrame(clf2.cv_results_)
# below shows the 10 best performers based on the TEST data (not the TRAINING data)
CVres[['rank_test_score',
       'mean_test_score',
       'mean_train_score',
       'param_hidden_layer_sizes',
       'param_activation',
       'param_alpha']][CVres['rank_test_score']<=10]




# EXAMPLE 2: Uses for loops
# Assumes you have all your data, both training and test data, in (X, y)
# X = input data, y = output/target data
X = 0
y = 0

neurons = [(10,), (50,), (100,), (150,), (200,),
           (100,100,), (100,200,), (200,100,)]
train_identity = []; test_identity = []
train_relu = []; test_relu = [] 
train_log = []; test_log = []; 
train_tanh = []; test_tanh = []; 

for i in neurons:
    print(i)
    cv_results = cross_validate(MLPRegressor(hidden_layer_sizes=(i,), activation='identity'),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
    train_identity.append(train_rmse)
    test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
    test_identity.append(test_rmse)

for i in neurons:
    print(i)
    cv_results = cross_validate(MLPRegressor(hidden_layer_sizes=(i,), activation='relu'),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
    train_relu.append(train_rmse)
    test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
    test_relu.append(test_rmse)
    
for i in neurons:
    print(i)
    cv_results = cross_validate(MLPRegressor(hidden_layer_sizes=(i,), activation='logistic'),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
    train_log.append(train_rmse)
    test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
    test_log.append(test_rmse)

for i in neurons:
    print(i)
    cv_results = cross_validate(MLPRegressor(hidden_layer_sizes=(i,), activation='tanh'),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
    train_tanh.append(train_rmse)
    test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
    test_tanh.append(test_rmse)

fig, ax = plt.subplots()
ax.plot(neurons,test_identity, label='No Activation Function Test RMSE')
ax.plot(neurons,train_identity, label='No Activation Function Train RMSE')
ax.plot(neurons,test_relu, label='Relu Test RMSE')
ax.plot(neurons,train_relu, label='Relu Train RMSE')
ax.plot(neurons,test_log, label='Logistic Test RMSE')
ax.plot(neurons,train_log, label='Logistic Train RMSE')
ax.plot(neurons,test_tanh, label='Tanh Test RMSE')
ax.plot(neurons,train_tanh, label='Tanh Train RMSE')
ax.legend(loc='best')
plt.xlabel("Number of neurons"); plt.ylabel("Average RMSE"); plt.title("Neural Network Regression model")
plt.show()

# plots for best result: change best_est to your best result
cv_results = cross_validate(MLPRegressor(hidden_layer_sizes=(200,), activation='relu'),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
print(np.sqrt(cv_results['test_score']*(-1.)))
best_est = cv_results['estimator'][9] # change best_est to your best result
y_pred = best_est.predict(X)
x = np.arange(len(y))+1

fig, ax = plt.subplots()
ax.scatter(y,y_pred, 'ro', label='Fitted')
ax.plot(y,y, 'bo', label='True')
ax.legend(loc='best')
plt.xlabel("True values"); 
plt.ylabel("Fitted Values"); 
plt.title("Neural Network Regression Fitted Values VS. True Values")
plt.show()

res = y - y_pred
fig, ax = plt.subplots()
ax.scatter(y_pred, res, 'ro', label='Residual')
ax.plot(y_pred, np.zeros_like(y_pred), 'bo')
ax.legend(loc='best')
plt.xlabel("Fitted Values"); 
plt.ylabel("Residuals"); 
plt.title("Neural Network Regression Residuals VS. Fitted Values")
plt.show()

# fig, ax = plt.subplots()
# ax.plot(x,y, 'ro', label='True')
# ax.plot(x,y_pred, 'bo', label='Fitted')
# ax.legend(loc='best')
# plt.xlabel("x"); 
# plt.ylabel("Y"); 
# plt.title("Neural Network Regression Fitted VS True")
# plt.show()

# res = y - y_pred
# fig, ax = plt.subplots()
# ax.plot(x,res, 'ro', label='Residual')
# ax.plot(x,y_pred, 'bo', label='Fitted')
# ax.legend(loc='best')
# plt.xlabel("x"); 
# plt.ylabel("Y"); 
# plt.title("Neural Network Regression Fitted VS Residuals")
# plt.show()
