# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:05:04 2020

@author: eric
"""
#%%
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import datetime
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
#from sklearn import svm
from sklearn.metrics import make_scorer,confusion_matrix
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
import xgboost as xgb
from scipy.stats import uniform, randint
#from xgboost import plot_importance
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns

from sklearn.metrics import accuracy_score

from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.neighbors import LocalOutlierFactor
from gensim.models import word2vec;print("FAST_VERSION", word2vec.FAST_VERSION)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import KNNImputer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, f1_score, precision_recall_curve, precision_score,recall_score
#from sklearn.metrics import recall_score, average_precision_score, auc
from catboost import CatBoostClassifier
import scikitplot as skplt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#%%

document_fill = pd.read_csv(r"C:\Users\eric\Desktop\CrunchBase_ML_Project\sepa_do.csv",header=0,encoding="utf8")

#%%

# document_fill["IT_Spend"] = document_fill["IT_Spend"].astype(float)
# document_fill["IT_Spend"].describe()

# for i in range(len(document_fill)):
#     if document_fill["IT_Spend"][i] >= 163970.969946:
#         document_fill["IT_Spend"][i] = 1
#     else:
#         document_fill["IT_Spend"][i] = 0
#%%
# document_fill["Estimated_Revenue_Range"] =document_fill["Estimated_Revenue_Range"].astype('category')
# document_fill["Last_Equity_Funding_Type"] =document_fill["Last_Equity_Funding_Type"].astype('category')
# document_fill["Visit_Duration"] = document_fill["Visit_Duration"].astype(float)
# document_fill["Visit_Duration"] = document_fill["Visit_Duration"].astype(float)

document_fill=document_fill.drop(["Total Equity Funding Amount","Last Funding Date","Number_of_Events","IT_Spend"],axis=1)
document_fill.info()
#%%


document_fill['Estimated_Revenue_Range'].replace(to_replace=['0', 'Less than $1M', '$1M to $10M','$10M to $50M','$50M to $100M','$100M to $500M','$500M to $1B','$1B to $10B','$10B+'], value=[1, 2, 3,4,5,6,7,8,9], inplace=True)

document_fill["Last_Equity_Funding_Type"].replace(to_replace=['Angel', 'Corporate Round', 'Equity Crowdfunding','Initial Coin Offering','Post-IPO Equity','Pre-Seed','Private Equity','Seed','Series A','Series B','Series C','Series D','Series E','Series F','Series G','Series H','Undisclosed','Venture - Series Unknown'], value=[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], inplace=True)


document_fill["Last_Equity_Funding_Type"] = document_fill["Last_Equity_Funding_Type"].astype(float)
# document_fill['Estimated_Revenue_Range'] = document_fill['Estimated_Revenue_Range'].astype('category')
# document_fill["Last_Equity_Funding_Type"] = document_fill["Last_Equity_Funding_Type"].astype('category')


# document_fill = pd.concat([document_fill.drop('Estimated_Revenue_Range', axis=1), pd.get_dummies(document_fill['Estimated_Revenue_Range'],prefix="Estimated_Revenue_Range")], axis=1)
# document_fill = pd.concat([document_fill.drop('Last_Equity_Funding_Type', axis=1), pd.get_dummies(document_fill['Last_Equity_Funding_Type'],prefix="Last_Equity_Funding_Type")], axis=1)



#%%
#impute miising values with machine learning method
# from sklearn.ensemble import RandomForestRegressor
# data_1=document_fill[["Number_of_Lead_Investors","Estimated_Revenue_Range","Total_Funding_Amount","Last_funding_to_date","Last_Equity_Funding_Type","Last_Equity_Funding_Amount"]]

# test_data = data_1[data_1["Number_of_Lead_Investors"].isnull()]
# data_1.dropna(inplace=True)

# y_train = data_1["Number_of_Lead_Investors"]
# X_train = data_1.drop("Number_of_Lead_Investors", axis=1)
# X_test = test_data.drop("Number_of_Lead_Investors", axis=1)

# model = RandomForestRegressor(n_jobs=-1)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

#%%

document_fill.isna().sum()
document_fill["IPO_Status"] = document_fill["IPO_Status"].astype("category")
#document_fill.to_csv(r"C:\Users\eric\Desktop\cb_ml\sepa_do1.csv",index=False,encoding='utf8')

#%%
####
X=document_fill.loc[:, document_fill.columns != 'IPO_Status']
y=document_fill['IPO_Status'] 

document_fill.info()
document_fill.isna().sum()


imputer = KNNImputer(n_neighbors=5)
X_tech = pd.DataFrame(imputer.fit_transform(X[["Patents_Granted","Active_Tech_Count"]]),columns = ["Patents_Granted","Active_Tech_Count"])

X_merchan = pd.DataFrame(imputer.fit_transform(X[["Total_Products_Active","Trademarks_Registered"]]),columns = ["Total_Products_Active","Trademarks_Registered"])

X_oppur = pd.DataFrame(imputer.fit_transform(X[["Company_Age","LOF_Score","Last_Equity_Funding_Type"]]),columns = ["Company_Age","LOF_Score","Last_Equity_Funding_Type"])

X_fund = pd.DataFrame(imputer.fit_transform(X[["Last_funding_to_date","Last_Equity_Funding_Amount","Number_of_Lead_Investors","Estimated_Revenue_Range","Total_Funding_Amount"]]),columns = ["Last_funding_to_date","Last_Equity_Funding_Amount","Number_of_Lead_Investors","Estimated_Revenue_Range","Total_Funding_Amount"])

X_pop = pd.DataFrame(imputer.fit_transform(X[["Average_Visits","Visit_Duration","Number_of_Articles","Headquarters_Regions"]]),columns = ["Average_Visits","Visit_Duration","Number_of_Articles","Headquarters_Regions"])

X = pd.concat([X_tech, X_merchan,X_oppur,X_fund,X_pop],axis=1)

#%%
########################################################################
#Oversampling
########################################################################
X_variable=X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

print("Before OverSampling, counts of label '1': {}".format(sum(y_test==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_test==0)))


plt.hist(y_train)
plt.draw()



#%%
########################################################################
#Modeling
########################################################################
#%%

#%%

def model_metrics(y_test,y_pred):
    acc = round( metrics.accuracy_score(y_test, y_pred) , 3)
    print( 'Accuracy of  model : ', acc )
    
    
    f1=round(f1_score(y_test, y_pred.round(), average='weighted'),3)
    roc_score=np.round(roc_auc_score(y_test, y_pred.round(), average='weighted'), decimals=4)
    recall = np.round(recall_score(y_test, y_pred.round(), average='weighted'), decimals=3)
    precis= np.round(precision_score(y_test, y_pred.round(), average='weighted'), decimals=3)
    print('f1 score: ', f1)
    print('auc  score: ', roc_score)
    print('recall score: ', recall)
    print('precision  score: ', precis)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred.round())

def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8, 6)) 
    plt.plot(fpr, tpr, linewidth=2, label=label, color='red')
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.title('Receiver Operating Curve(ROC)', fontsize=18)
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    #plt.legend(True)
    plt.grid(True)
    plt.show()
    
#%%
# Create a Logistic regression classifier
logreg = LogisticRegression()

grid={"C":np.logspace(-4, 4, 50), "penalty":["l1","l2"]}
grid_log=GridSearchCV(logreg,grid,cv=10)
# Train the model using the training sets 
grid_log.fit(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ",grid_log.best_params_)
print("accuracy :",grid_log.best_score_)

logreg_2 = LogisticRegression(C=0.0001,penalty='l2')
logreg_2.fit(X_train, y_train)
# Prediction on test data
y_pred_LF = logreg_2.predict(X_test)
model_metrics(y_test,y_pred_LF)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_LF) 
plot_roc_curve(fpr, tpr)

#%%
# Create a Decision tree classifier model
clf = DecisionTreeClassifier()
# Hyperparameter Optimization
parameters = {'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [5, 10,15,20,25,30], 
              'min_samples_split': [2, 3, 50, 100],
              'min_samples_leaf': [1, 5, 8, 10]
             }

# Run the grid search
grid_obj = GridSearchCV(clf, parameters,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Train the model using the training sets 
clf.fit(X_train, y_train)
# Prediction on test set
y_pred_dt= clf.predict(X_test)
# Calculating the accuracy
model_metrics(y_test,y_pred_dt)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_dt) 
plot_roc_curve(fpr, tpr)
#%%
#tree visualization
from sklearn.tree import export_graphviz

#export_graphviz(clf, out_file='tree.dot')
export_graphviz(clf, out_file='tree_WHITE_wine_dt.dot', 
                feature_names = X.columns,
                class_names = ["0","1"],
                max_depth = 2, # 표현하고 싶은 최대 depth
                precision = 2, # 소수점 표기 자릿수
                filled = True, # class별 color 채우기
                rounded=True, # 박스의 모양을 둥글게
               )


#%%
# Create a Random Forest Classifier
rf = RandomForestClassifier()

# Hyperparameter Optimization
parameters = {'n_estimators': [10,20,30], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [5, 10,15], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
             }

# Run the grid search
grid_obj = GridSearchCV(rf, parameters,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the rf to the best combination of parameters
rf = grid_obj.best_estimator_

# Train the model using the training sets 
rf.fit(X_train, y_train)
estimator = rf.estimators_[3]
# Prediction on test data
y_pred_rf = rf.predict(X_test)

# Calculating the accuracy
model_metrics(y_test,y_pred_rf)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf) 
plot_roc_curve(fpr, tpr)

featureImpList= []
for feat, importance in zip(X_train.columns, rf.feature_importances_):  
    temp = [feat, importance*100]
    featureImpList.append(temp)

fT_df = pd.DataFrame(featureImpList, columns = ['Feature', 'Importance'])
print (fT_df.sort_values('Importance', ascending = False))
#%%
svc = StandardScaler()
X_train_scaled = svc.fit_transform(X_train)
X_test_scaled = svc.transform(X_test)

#%%
#ligthgbm
params = {
    'application': 'binary', # for binary classification
#     'num_class' : 1, # used for multi-classes
    'boosting': 'gbdt', # traditional gradient boosting decision tree
    'num_iterations': 100, 
    'learning_rate': 0.05,
    'num_leaves': 62,
    'device': 'cpu', # you can use GPU to achieve faster learning
    'max_depth': 15, # <0 means no limit
    'max_bin': 510, # Small number of bins may reduce training accuracy but can deal with over-fitting
    'lambda_l1': 5, # L1 regularization
    'lambda_l2': 10, # L2 regularization
    'metric' : 'binary_error',
    'subsample_for_bin': 200, # number of samples for constructing bins
    'subsample': 1, # subsample ratio of the training instance
    'colsample_bytree': 0.8, # subsample ratio of columns when constructing the tree
    'min_split_gain': 0.5, # minimum loss reduction required to make further partition on a leaf node of the tree
    'min_child_weight': 1, # minimum sum of instance weight (hessian) needed in a leaf
    'min_child_samples': 5# minimum number of data needed in a leaf
}

mdl = lgb.LGBMClassifier(boosting_type= 'gbdt', 
          objective = 'binary', 
          n_jobs = -1, 
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'], 
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'], 
          min_split_gain = params['min_split_gain'], 
          min_child_weight = params['min_child_weight'], 
          min_child_samples = params['min_child_samples'])

x_train = svc.fit_transform(X_train)
x_test = svc.transform(X_test)
train_ds = lgb.Dataset(x_train, label = y_train) 
test_ds = lgb.Dataset(x_test, label = y_test) 

gridParams = {
    'learning_rate': [0.005, 0.01],
    'n_estimators': [10,20,30],
    'num_leaves': [6,8,12,16], 
    'boosting_type' : ['gbdt', 'dart'],
    'objective' : ['binary'],
    'max_bin':[255, 510], 
    'random_state' : [500],
    'colsample_bytree' : [0.64, 0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }
lgb_grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)
# Run the grid
lgb_grid.fit(X_train, y_train)

light_grid = lgb_grid.best_estimator_
# Train the model using the training sets 
light_grid.fit(X_train, y_train)

# Prediction on test data
y_pred_light = light_grid.predict(X_test)

model_metrics(y_test,y_pred_light)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_light) 
plot_roc_curve(fpr, tpr)

ax = lgb.plot_importance(light_grid, max_num_features=18, figsize=(10,10))
plt.show()
#%%
# import graphviz 
# export_graphviz(estimator, out_file='tree_rf.dot', 
#                 feature_names = X.columns,
#                 class_names = ["0","1"],
#                 max_depth = 3, # 표현하고 싶은 최대 depth
#                 precision = 3, # 소수점 표기 자릿수
#                 filled = True, # class별 color 채우기
#                 rounded=True, # 박스의 모양을 둥글게
#                )

# dot_data = lgb.create_tree_digraph(light_grid, tree_index = 1,show_info=['split_gain'])

# graph = graphviz.Source(dot_data)  
# graph 
#%%
#xgb
xgb_model = xgb.XGBClassifier()


gridParams = {
    'learning_rate': [0.005, 0.01],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'min_child_weight': [1, 5, 10],
    'n_estimators': [10,20,30],
    'num_leaves': [6,8,12,16], 
    'boosting_type' : ['gbdt', 'dart'],
    'max_bin':[255, 510], 
    'random_state' : [500],
    'colsample_bytree' : [0.64, 0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }

search = GridSearchCV(xgb_model,gridParams,cv=3, verbose=1, n_jobs=-1)

search.fit(X_train, y_train)

xgb_grid = search.best_estimator_

y_pred_xgb= xgb_grid.predict(X_test)

# Calculating the accuracy
model_metrics(y_test,y_pred_xgb)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_xgb) 
plot_roc_curve(fpr, tpr)

#xgb feature importance

fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(xgb_grid, max_num_features=10, height=0.5, ax=ax,importance_type='gain')
plt.show()

#%%
#adaboost

n_estimators = [10,20,30];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=42)
learning_r = [0.005, 0.01]

parameters = {'n_estimators':n_estimators,
              'learning_rate':learning_r
              
        }
grid = GridSearchCV(AdaBoostClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.
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
model_metrics(y_test,y_pred_ada)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_ada) 
plot_roc_curve(fpr, tpr)

#%%
clf = CatBoostClassifier()
params = {'iterations': [10],
          'depth': [12,15],
          'loss_function': ['Logloss', 'CrossEntropy'],
          'l2_leaf_reg': [5],
          'leaf_estimation_iterations': [10],
#           'eval_metric': ['Accuracy'],
#           'use_best_model': ['True'],
          'logging_level':['Silent'],
          'random_seed': [42]
         }
scorer = make_scorer(accuracy_score)
clf_grid = GridSearchCV(estimator=clf, param_grid=params, scoring=scorer, cv=5,n_jobs=-1)
clf_grid.fit(X_train_res, y_train_res.ravel())


cat_grid = clf_grid.best_estimator_
# Train the model using the training sets 
cat_grid.fit(X_train_res, y_train_res.ravel())

# Prediction on test data
y_pred_cat = cat_grid.predict(X_test)

model_metrics(y_test,y_pred_cat)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_cat) 
plot_roc_curve(fpr, tpr)

#%%
#%%
parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01],
    # "min_samples_split": np.linspace(0.1, 0.5, 12),
    # "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[12,15],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.7,0.75],
    "n_estimators":[30]
    }

clf_gb = GridSearchCV(GradientBoostingClassifier(), parameters,n_jobs=-1)
clf_gb.fit(X_train, y_train)


clf_gb_best = clf_gb.best_estimator_
clf_gb_best.fit(X_train, y_train)
# print(clf.score(X_train_res, y_train_res.ravel()))
# print(clf.best_params_)

y_pred_gb= clf_gb_best.predict(X_test)

model_metrics(y_test,y_pred_gb)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_gb) 
plot_roc_curve(fpr, tpr)
