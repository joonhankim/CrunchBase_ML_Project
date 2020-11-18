# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 05:05:41 2020

@author: eric
"""
########################################################################
# Import packages
########################################################################
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
#################

#Data load


################
#cb_rank삭제

document=pd.read_csv(r"C:\Users\eric\Desktop\CrunchBase_ML_Project\2010_2019_healthcare.csv",header=0,encoding="ISO-8859-1")

document.info()
document.isna().sum()
document.head()
len(document)


#%%
pd.set_option('float_format', '{:f}'.format)

#%%
lof_document = document[['Full Description']]

cb_healthcare=lof_document.rename(columns={'Full Description': 'Text'})
cb_healthcare.Text=cb_healthcare.Text.astype(str)
#cb_healthcare=cb_healthcare.fillna(0)
#doc2vec
lmtzr = WordNetLemmatizer()
w = re.compile("\w+",re.I)

def label_sentences(df):
    labeled_sentences = []
    for index, datapoint in df.iterrows():
        tokenized_words = re.findall(w,datapoint["Text"].lower())
        labeled_sentences.append(LabeledSentence(words=tokenized_words, tags=['SENT_%s' %index]))
    return labeled_sentences

def train_doc2vec_model(labeled_sentences):
    model = Doc2Vec(alpha=0.025, min_alpha=0.025,workers=-1)
    model.build_vocab(labeled_sentences)
    for epoch in range(10):
        model.train(labeled_sentences,epochs=model.iter,total_examples=model.corpus_count)
        model.alpha -= 0.002 
        model.min_alpha = model.alpha
    
    return model

sen = label_sentences(cb_healthcare)
model = train_doc2vec_model(sen)


vector_list=[]
for i in range(len(cb_healthcare)):
    vector_list.append(model.docvecs[i])

X=vector_list

df_x = pd.DataFrame(X)
clf_lof = LocalOutlierFactor(n_neighbors=70,metric='euclidean',n_jobs=-1)
y_pred = clf_lof.fit_predict(df_x)
y_pred_score = clf_lof._decision_function(df_x)

X_scores = -(clf_lof.negative_outlier_factor_)

lofScores_10 = pd.DataFrame(X_scores,columns = ['LOF_scores'])
#lofScores_10['N/M']=patent['기업이름']
sort = lofScores_10.sort_values(["LOF_scores"],ascending=[False])
sort=sort.reset_index(drop= True)
#sorts2 = np.array(sort)

#%%
X_embedded = TSNE(n_components=2,n_jobs=-1).fit_transform(X)
X_embedded[:, 0]
#%%

fig = plt.figure(figsize = (8,5))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
plt.title("Local Outlier Factor (LOF)",fontsize = 20)
#plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
b = plt.scatter(X_embedded[21:, 0], X_embedded[21:, 1], c='blue',edgecolor='k', s=20)
a = plt.scatter(X_embedded[:20,0 ], X_embedded[:20, 1], c='red',edgecolor='k', s=30)

#plt.axis('tight')
#plt.xlim((-1, 2))
#plt.ylim((-2, 2))
plt.legend([a, b],
           ["abnormal observations",
            
            "normal observations"],
           loc="upper left")
plt.show()
#%%
#sort["LOF_scores"][824]

#for i in range(len(sort)):
    #if sort["LOF_scores"][i] > 1.0371315461641502:
       #sort["LOF_scores"][i] = 1
    #else: 
        #sort["LOF_scores"][i] = 0
        
#document["LOF Score"]= sort
#%%


document["LOF_Score"]= X_scores

document["LOF_Score"][1]

print(type(document["LOF_Score"]))
document=document.drop(["Full Description"],axis=1)




#%%
########################################################################
# calculate company age
########################################################################

document['Founded Date'] = pd.to_datetime(document['Founded Date'],errors="coerce")
document['Founded Date'] = pd.to_datetime(document['Founded Date']).dt.date
type(document['Founded Date'][0])
d = datetime.date(2020, 11, 5)
document["Company_Age"]=d-document['Founded Date']
document["Company_Age"].head(3)
document["Company_Age"]=document["Company_Age"].astype('timedelta64[D]')
document["Company_Age"].head(3)

# calculate last funding to date
document['Last Funding Date'] = pd.to_datetime(document['Last Funding Date'],errors="coerce")
document['Last Funding Date'] = pd.to_datetime(document['Last Funding Date'],errors="coerce").dt.date
document["Last_funding_to_date"]=d-document["Last Funding Date"]
document["Last_funding_to_date"]=document["Last_funding_to_date"].astype('timedelta64[D]')
document["Last_funding_to_date"][0]

document=document.drop(['Founded Date'], axis=1)
document.isna().sum()
document.info()



#document_fill = document.fillna(0)


###remove , in excel####
#%%
document.to_csv(r"C:\Users\eric\Desktop\CrunchBase_ML_Project\sepa_do.csv",index=False,encoding='utf8')

document_fill = pd.read_csv(r"C:\Users\eric\Desktop\CrunchBase_ML_Project\sepa_do.csv",header=0,encoding="utf8")
#%%
#document_fill=document_fill.drop("CB_Rank",axis=1)
document_fill["IPO_Status"] = document_fill["IPO_Status"].astype("category")
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

document_fill=document_fill.drop(["Total Equity Funding Amount","Last Funding Date"],axis=1)
document_fill.info()
#%%


document_fill['Estimated_Revenue_Range'].replace(to_replace=['0', 'Less than $1M', '$1M to $10M','$10M to $50M','$50M to $100M','$100M to $500M','$500M to $1B','$1B to $10B','$10B+'], value=[1, 2, 3,4,5,6,7,8,9], inplace=True)

document_fill["Last_Equity_Funding_Type"].replace(to_replace=['Angel', 'Corporate Round', 'Equity Crowdfunding','Initial Coin Offering','Post-IPO Equity','Pre-Seed','Private Equity','Seed','Series A','Series B','Series C','Series D','Series E','Series F','Series G','Series H','Undisclosed','Venture - Series Unknown'], value=[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], inplace=True)


document_fill["Last_Equity_Funding_Type"] = document_fill["Last_Equity_Funding_Type"].astype(float)
# document_fill['Estimated_Revenue_Range'] = document_fill['Estimated_Revenue_Range'].astype('category')
# document_fill["Last_Equity_Funding_Type"] = document_fill["Last_Equity_Funding_Type"].astype('category')


# document_fill = pd.concat([document_fill.drop('Estimated_Revenue_Range', axis=1), pd.get_dummies(document_fill['Estimated_Revenue_Range'],prefix="Estimated_Revenue_Range")], axis=1)
# document_fill = pd.concat([document_fill.drop('Last_Equity_Funding_Type', axis=1), pd.get_dummies(document_fill['Last_Equity_Funding_Type'],prefix="Last_Equity_Funding_Type")], axis=1)



document_fill.info()

imputer = KNNImputer(n_neighbors=5)
document_fill = pd.DataFrame(imputer.fit_transform(document_fill),columns = document_fill.columns)
document_fill.isna().sum()
document_fill["IPO_Status"] = document_fill["IPO_Status"].astype("category")
#document_fill.to_csv(r"C:\Users\eric\Desktop\cb_ml\sepa_do1.csv",index=False,encoding='utf8')

#%%
####
X=document_fill.loc[:, document_fill.columns != 'IPO_Status']
y=document_fill['IPO_Status'] 
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

sm = SMOTE(random_state=2,k_neighbors=5)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


plt.hist(y_train_res)
plt.draw()



#%%

########################################################################
#EDA
########################################################################


X_train_res.hist(figsize=(15,15)) 
#np.log(X_train_res["IPqwery - Patents Granted"]).plot.hist
plt.tight_layout()
plt.draw()

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(42)


# Compute the correlation matrix
corr = X_train_res.corr()


# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.7, cbar_kws={"shrink": .5})

#%%
sns.distplot(X_train_res, hist=False, kde_kws={'clip': (0.0, 100)})
plt.show()
#%%
sns.distplot(X_train_res["Total Funding Amount"])
plt.show()
#%%
sns.countplot(data = X_train_res, x = 'Total Funding Amount')
plt.show()
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
    roc_score=np.round(roc_auc_score(y_test, y_pred_LF.round(), average='weighted'), decimals=3)
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
grid_log.fit(X_train_res, y_train_res.ravel())
print("tuned hpyerparameters :(best parameters) ",grid_log.best_params_)
print("accuracy :",grid_log.best_score_)

logreg_2 = LogisticRegression(C=0.0001,penalty='l2')
logreg_2.fit(X_train_res, y_train_res.ravel())
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
grid_obj = grid_obj.fit(X_train_res, y_train_res.ravel())

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Train the model using the training sets 
clf.fit(X_train_res, y_train_res.ravel())
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
export_graphviz(clf, out_file='tree_dong_dt.dot', 
                feature_names = X.columns,
                class_names = ["0","1"],
                max_depth = 2, # 표현하고 싶은 최대 depth
                precision = 2, # 소수점 표기 자릿수
                filled = True, # class별 color 채우기
                rounded=True, # 박스의 모양을 둥글게
               )
#%%
#%%
parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[6,9,12,15],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10,20,30]
    }

clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

clf.fit(X_train_res, y_train_res.ravel())
print(clf.score(X_train_res, y_train_res.ravel()))
print(clf.best_params_)

y_pred_gb= clf.predict(X_test)

model_metrics(y_test,y_pred_gb)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_gb) 
plot_roc_curve(fpr, tpr)
#%%
# Create a Random Forest Classifier
rf = RandomForestClassifier()

# Hyperparameter Optimization
parameters = {'n_estimators': [10,20,30], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [5, 10,15,20,25,30], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
             }

# Run the grid search
grid_obj = GridSearchCV(rf, parameters,n_jobs=-1)
grid_obj = grid_obj.fit(X_train_res, y_train_res.ravel())

# Set the rf to the best combination of parameters
rf = grid_obj.best_estimator_

# Train the model using the training sets 
rf.fit(X_train_res, y_train_res.ravel())
estimator = rf.estimators_[3]
# Prediction on test data
y_pred_rf = rf.predict(X_test)

# Calculating the accuracy
model_metrics(y_test,y_pred_rf)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf) 
plot_roc_curve(fpr, tpr)

featureImpList= []
for feat, importance in zip(X_train_res.columns, rf.feature_importances_):  
    temp = [feat, importance*100]
    featureImpList.append(temp)

fT_df = pd.DataFrame(featureImpList, columns = ['Feature', 'Importance'])
print (fT_df.sort_values('Importance', ascending = False))
#%%
svc = StandardScaler()
X_train_scaled = svc.fit_transform(X_train_res)
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
    'max_depth': -1, # <0 means no limit
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

x_train = svc.fit_transform(X_train_res)
x_test = svc.transform(X_test)
train_ds = lgb.Dataset(x_train, label = y_train_res.ravel()) 
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
lgb_grid.fit(X_train_res, y_train_res.ravel())

light_grid = lgb_grid.best_estimator_
# Train the model using the training sets 
light_grid.fit(X_train_res, y_train_res.ravel())

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
    "max_depth": -1, # default 3
    "n_estimators": [10,20,30], # default 100
    "subsample": uniform(0.6, 0.4)
}

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

search.fit(X_train_res, y_train_res.ravel())

report_best_scores(search.cv_results_, 1)

y_pred_xgb= search.predict(X_test)

# Calculating the accuracy
model_metrics(y_test,y_pred_xgb)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_xgb) 
plot_roc_curve(fpr, tpr)
#%%
#adaboost

n_estimators = [10,20,30];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=42)
learning_r = uniform(0.03, 0.3)

parameters = {'n_estimators':n_estimators,
              'learning_rate':learning_r
              
        }
grid = GridSearchCV(AdaBoostClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.
                                     ),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
            
grid.fit(X_train_res, y_train_res.ravel()) 

adaBoost_grid = grid.best_estimator_
# Train the model using the training sets 
adaBoost_grid.fit(X_train_res, y_train_res.ravel())

# Prediction on test data
y_pred_ada = adaBoost_grid.predict(X_test)

# Calculating the accuracy
model_metrics(y_test,y_pred_ada)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_ada) 
plot_roc_curve(fpr, tpr)

#%%
clf = CatBoostClassifier()
params = {'iterations': [500],
          'depth': [6,9,12,15],
          'loss_function': ['Logloss', 'CrossEntropy'],
          'l2_leaf_reg': np.logspace(-20, -19, 3),
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
# plt.figure(figsize=(15, 4))
# plt.plot(['Logistic','AdaBoost',"Decision Tree","RandonForest","Support Vector Machines","LightGBM","Xgboost"], [acc_logreg, acc_ada,acc_dt, acc_rf, acc_svm, acc_lgb,acc_xgb], 'ro')

# plt.show()
#%%
