# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 06:38:18 2020

@author: eric
"""

# import packages
import pandas as pd
import datetime
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.metrics import f1_score
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

document_inter=document.interpolate()
document_inter=document_inter.fillna(0)
X=document_inter.loc[:, document_inter.columns != 'IPO Status']
y=document_inter['IPO Status'] 

#Oversampling


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


plt.hist(y_train_res)
plt.show()

# Create a Logistic regression classifier
logreg = LogisticRegression()

# Train the model using the training sets 
logreg.fit(X_train_res, y_train_res.ravel())
# Prediction on test data
y_pred = logreg.predict(X_test)
acc_logreg = round( metrics.accuracy_score(y_test, y_pred) , 3)
print( 'Accuracy of Logistic Regression model : ', acc_logreg )
mse_logreg = mean_squared_error(y_test, y_pred)
r2_logreg= r2_score(y_test, y_pred)
f1_logreg=f1_score(y_test, y_pred, average='weighted')
print('Mean squared error: ', mse_logreg)
print('R2 score: ', r2_logreg)
# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(X_train_res, y_train_res.ravel())

# Prediction on test set
y_pred = model.predict(X_test)
# Calculating the accuracy
acc_nb = round( metrics.accuracy_score(y_test, y_pred) , 3 )
print( 'Accuracy of Gaussian Naive Bayes model : ', acc_nb )
mse_nb = mean_squared_error(y_test, y_pred)
r2_nb= r2_score(y_test, y_pred)
f1_nb=f1_score(y_test, y_pred, average='weighted')
print('Mean squared error: ', mse_nb)
print('R2 score: ', r2_nb)
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
grid_obj = grid_obj.fit(X_train_res, y_train_res.ravel())

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Train the model using the training sets 
clf.fit(X_train_res, y_train_res.ravel())
# Prediction on test set
y_pred = clf.predict(X_test)
# Calculating the accuracy
acc_dt = round( metrics.accuracy_score(y_test, y_pred) , 3 )
print( 'Accuracy of Decision Tree model : ', acc_dt )
mse_dt = mean_squared_error(y_test, y_pred)
r2_dt= r2_score(y_test, y_pred)
f1_dt=f1_score(y_test, y_pred, average='weighted')
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
grid_obj = grid_obj.fit(X_train_res, y_train_res.ravel())

# Set the rf to the best combination of parameters
rf = grid_obj.best_estimator_

# Train the model using the training sets 
rf.fit(X_train_res, y_train_res.ravel())

# Prediction on test data
y_pred = rf.predict(X_test)

# Calculating the accuracy
acc_rf = round( metrics.accuracy_score(y_test, y_pred)  , 3 )
print( 'Accuracy of Random Forest model : ', acc_rf )
mse_rf = mean_squared_error(y_test, y_pred)
r2_rf= r2_score(y_test, y_pred)
f1_rf=f1_score(y_test, y_pred, average='weighted')
print('Mean squared error: ', mse_rf)
print('R2 score: ', r2_rf)


featureImpList= []
for feat, importance in zip(X_train_res.columns, rf.feature_importances_):  
    temp = [feat, importance*100]
    featureImpList.append(temp)

fT_df = pd.DataFrame(featureImpList, columns = ['Feature', 'Importance'])
print (fT_df.sort_values('Importance', ascending = False))


sc = StandardScaler()
X_train = sc.fit_transform(X_train_res)
X_test = sc.transform(X_test)

# Create a Support Vector Classifier
svc = svm.SVC()

# Train the model using the training sets 
svc.fit(X_train,y_train_res.ravel())

# Prediction on test data
y_pred = svc.predict(X_test)

# Calculating the accuracy
acc_svm = round( metrics.accuracy_score(y_test, y_pred) , 3 )
print( 'Accuracy of SVM model : ', acc_svm )
mse_svm = mean_squared_error(y_test, y_pred)
r2_svm= r2_score(y_test, y_pred)
f1_svm=f1_score(y_test, y_pred, average='weighted')
print('Mean squared error: ', mse_svm)
print('R2 score: ', r2_svm)
#ligthgbm
x_train = sc.fit_transform(X_train_res)
x_test = sc.transform(X_test)
train_ds = lgb.Dataset(x_train, label = y_train_res.ravel()) 
test_ds = lgb.Dataset(x_test, label = y_test) 

params = {'learning_rate': 0.01, 
          'max_depth': 16, 
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'metric': 'binary_logloss', 
          'is_training_metric': True, 
          'num_leaves': 144}

lgb_model = lgb.train(params, train_ds, 1000, test_ds, verbose_eval=100, early_stopping_rounds=100)
predict_train = lgb_model.predict(X_train_res)
predict_test = lgb_model.predict(x_test)

# Calculating the accuracy
acc_lgb = round( metrics.accuracy_score(predict_test.round(), y_test) , 3 )
print( 'Accuracy of LGB model : ', acc_lgb )
mse_lgb = mean_squared_error(y_test, y_pred)
r2_lgb= r2_score(y_test, y_pred)
f1_lgb=f1_score(y_test, y_pred, average='weighted')
print('Mean squared error: ', mse_lgb)
print('R2 score: ', r2_lgb)


ax = lgb.plot_importance(lgb_model, max_num_features=10, figsize=(10,10))
plt.show()