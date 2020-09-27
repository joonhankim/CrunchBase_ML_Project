# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:11:45 2020

@author: eric
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 05:05:41 2020

@author: eric
"""
# import packages
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import datetime
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
import xgboost as xgb
from scipy.stats import uniform, randint
#from xgboost import plot_importance
from sklearn.model_selection import StratifiedShuffleSplit

#Data load
document_healthcare=pd.read_csv(".\document_healthcare.csv",header=0,encoding='utf8')

#Data info
document_healthcare.info()
#data head
document_healthcare.head()

len(document_healthcare)

document_healthcare.isna().sum()

document=document_healthcare.drop(["Last Equity Funding Amount Currency (in USD)","Last Leadership Hiring Date","Number of Events","Apptopia - Downloads Last 30 Days","Apptopia - Number of Apps"],axis=1)

document.isna().sum()

# calculate company age
document['Founded Date'] = pd.to_datetime(document['Founded Date'],errors="coerce")
document['Founded Date'] = pd.to_datetime(document['Founded Date']).dt.date
type(document['Founded Date'][0])
d = datetime.date(2020, 7, 1)
document["Company Age"]=d-document['Founded Date']
document["Company Age"].head(3)
document["Company Age"]=document["Company Age"].astype('timedelta64[D]')
document["Company Age"].head(3)

# calculate last funding to date
document['Last Funding Date'] = pd.to_datetime(document['Last Funding Date'],errors="coerce")
document['Last Funding Date'] = pd.to_datetime(document['Last Funding Date'],errors="coerce").dt.date
document["Last funding to date"]=d-document["Last Funding Date"]
document["Last funding to date"]=document["Last funding to date"].astype('timedelta64[D]')
document["Last funding to date"][0]

document=document.drop(['Founded Date', 'Last Funding Date'], axis=1)
document.isna().sum()
document.info()

document_fill = document.fillna(0)

X=document_fill.loc[:, document_fill.columns != 'IPO Status']
y=document_fill['IPO Status'] 

#Oversampling
X_variable=X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)




plt.hist(y_train)
plt.show()


########################################################################
#Modeling
########################################################################


# Create a Logistic regression classifier
logreg = LogisticRegression()

grid={"C":np.logspace(-4, 4, 50), "penalty":["l1","l2"]}
grid_log=GridSearchCV(logreg,grid,cv=10)
# Train the model using the training sets 
grid_log.fit(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ",grid_log.best_params_)
print("accuracy :",grid_log.best_score_)

logreg_2 = LogisticRegression(C=1.7575106248547894,penalty="l2")
logreg_2.fit(X_train, y_train)
# Prediction on test data
y_pred_LF = logreg_2.predict(X_test)
acc_logreg = round( metrics.accuracy_score(y_test, y_pred_LF) , 3)
print( 'Accuracy of Logistic Regression model : ', acc_logreg )
mse_logreg = mean_squared_error(y_test, y_pred_LF)
r2_logreg= r2_score(y_test, y_pred_LF)
f1_logreg=f1_score(y_test, y_pred_LF, average='weighted')
roc_logreg=np.round(roc_auc_score(y_test, y_pred_LF, average='weighted'), decimals=4)
print('Mean squared error: ', mse_logreg)
print('R2 score: ', r2_logreg)





# Create a Decision tree classifier model
clf = DecisionTreeClassifier()
# Hyperparameter Optimization
parameters = {'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10, 50], 
              'min_samples_split': [2, 3, 50, 100],
              'min_samples_leaf': [1, 5, 8, 10]
             }

# Run the grid search
grid_obj = GridSearchCV(clf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Train the model using the training sets 
clf.fit(X_train, y_train)
# Prediction on test set
y_pred_dt= clf.predict(X_test)
# Calculating the accuracy
acc_dt = round( metrics.accuracy_score(y_test, y_pred_dt) , 3 )
print( 'Accuracy of Decision Tree model : ', acc_dt )
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt= r2_score(y_test, y_pred_dt)
f1_dt=f1_score(y_test, y_pred_dt, average='weighted')
roc_dt=np.round(roc_auc_score(y_test, y_pred_dt, average='weighted'), decimals=4)
print('Mean squared error: ', mse_dt)
print('R2 score: ', r2_dt)









# Create a Random Forest Classifier
rf = RandomForestClassifier()

# Hyperparameter Optimization
parameters = {'n_estimators': [4, 6, 9, 10, 15], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
             }

# Run the grid search
grid_obj = GridSearchCV(rf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the rf to the best combination of parameters
rf = grid_obj.best_estimator_

# Train the model using the training sets 
rf.fit(X_train, y_train)

# Prediction on test data
y_pred_rf = rf.predict(X_test)

# Calculating the accuracy
acc_rf = round( metrics.accuracy_score(y_test, y_pred_rf)  , 3 )
print( 'Accuracy of Random Forest model : ', acc_rf )
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf= r2_score(y_test, y_pred_rf)
f1_rf=f1_score(y_test, y_pred_rf, average='weighted')
roc_rf=np.round(roc_auc_score(y_test, y_pred_rf, average='weighted'), decimals=4)
print('Mean squared error: ', mse_rf)
print('R2 score: ', r2_rf)


featureImpList= []
for feat, importance in zip(X_train.columns, rf.feature_importances_):  
    temp = [feat, importance*100]
    featureImpList.append(temp)

fT_df = pd.DataFrame(featureImpList, columns = ['Feature', 'Importance'])
print (fT_df.sort_values('Importance', ascending = False))






svc = StandardScaler()
X_train_scaled = svc.fit_transform(X_train)
X_test_scaled = svc.transform(X_test)

clf_svc= svm.SVC()

params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(clf_svc,params,cv=5,n_jobs=-1)
grid.fit(X_train_scaled,y_train)

gird_svc=grid.best_estimator_
# Train the model using the training sets 
gird_svc.fit(X_train_scaled,y_train)

# Prediction on test data
y_pred_sc = gird_svc.predict(X_test_scaled)

# Calculating the accuracy
acc_svm = round( metrics.accuracy_score(y_test, y_pred_sc) , 3 )
print( 'Accuracy of SVM model : ', acc_svm )
mse_svm = mean_squared_error(y_test, y_pred_sc)
r2_svm= r2_score(y_test, y_pred_sc)
f1_svm=f1_score(y_test, y_pred_sc, average='weighted')
roc_svm=np.round(roc_auc_score(y_test, y_pred_sc, average='weighted'), decimals=4)
print('Mean squared error: ', mse_svm)
print('R2 score: ', r2_svm)






#ligthgbm
x_train = svc.fit_transform(X_train)
x_test = svc.transform(X_test)
train_ds = lgb.Dataset(x_train, label = y_train) 
test_ds = lgb.Dataset(x_test, label = y_test) 

params = {'learning_rate': 0.01, 
          'max_depth': 16, 
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'metric': 'binary_logloss', 
          'is_training_metric': True, 
          'num_leaves': 144}

lgb_model = lgb.train(params, train_ds, 1000, test_ds, verbose_eval=100, early_stopping_rounds=100)
predict_train = lgb_model.predict(X_train)
predict_test = lgb_model.predict(x_test)

# Calculating the accuracy
acc_lgb = round( metrics.accuracy_score(predict_test.round(), y_test) , 3 )
print( 'Accuracy of LGB model : ', acc_lgb )
mse_lgb = mean_squared_error(y_test, predict_test.round())
r2_lgb= r2_score(y_test, predict_test.round())
f1_lgb=f1_score(y_test, predict_test.round(), average='weighted')
roc_lgb=np.round(roc_auc_score(y_test, predict_test.round(), average='weighted'), decimals=4)
print('Mean squared error: ', mse_lgb)
print('R2 score: ', r2_lgb)


ax = lgb.plot_importance(lgb_model, max_num_features=10, figsize=(10,10))
plt.show()









#xgb

xgb_model = xgb.XGBRegressor()

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

search.fit(X_train, y_train)

report_best_scores(search.cv_results_, 1)

y_pred_xgb= search.predict(X_test)

# Calculating the accuracy
acc_xgb = round( metrics.accuracy_score(y_pred_xgb.round(), y_test) , 3 )
print( 'Accuracy of XGB model : ', acc_xgb )
mse_xgb = mean_squared_error(y_test, y_pred_xgb.round())
r2_xgb= r2_score(y_test, y_pred_xgb.round())
f1_xgb=f1_score(y_test, y_pred_xgb.round(), average='weighted')
roc_xgb=np.round(roc_auc_score(y_test, y_pred_xgb.round(), average='weighted'), decimals=4)
print('Mean squared error: ', mse_xgb)
print('R2 score: ', r2_xgb)

#xgb.plot_importance(search)


#adaboost
n_estimators = [100,140,145,150,160, 170,175,180,185];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=42)
learning_r = [0.1,1,0.01,0.5]

parameters = {'n_estimators':n_estimators,
              'learning_rate':learning_r
              
        }
grid = GridSearchCV(AdaBoostRegressor(base_estimator= None, ## If None, then the base estimator is a decision tree.
                                     ),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
            
grid.fit(X_train, y_train) 

adaBoost_grid = grid.best_estimator_
# Train the model using the training sets 
adaBoost_grid.fit(X_train, y_train)

# Prediction on test data
y_pred_ada = adaBoost_grid.predict(X_test)

# Calculating the accuracy
acc_ada = round( metrics.accuracy_score(y_pred_ada.round(), y_test) , 3 )
print( 'Accuracy of AdaBoost model : ', acc_ada )
mse_ada = mean_squared_error(y_test, y_pred_ada.round())
r2_ada= r2_score(y_test, y_pred_ada.round())
f1_ada=f1_score(y_test, y_pred_ada.round(), average='weighted')
roc_ada=np.round(roc_auc_score(y_test, y_pred_ada.round(), average='weighted'), decimals=4)
print('Mean squared error: ', mse_ada)
print('R2 score: ', r2_ada)






