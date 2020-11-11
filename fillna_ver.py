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
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import codecs
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
#%%
#################

#Data load


################

document_healthcare=pd.read_csv(r"C:\Users\eric\Desktop\cb_ml\2010_2019_healthcare.csv",header=0,encoding="ISO-8859-1")
#Data info
document_healthcare.info()

#data head
document_healthcare.head()


len(document_healthcare)


#%%
pd.set_option('float_format', '{:f}'.format)
#%%
#Check null

document=document_healthcare.drop(["Number of Events","Apptopia - Downloads Last 30 Days","Apptopia - Number of Apps","Unnamed: 27","Unnamed: 28","Unnamed: 29","Unnamed: 30","Unnamed: 31","Unnamed: 32","Unnamed: 28","Unnamed: 29","Unnamed: 30","Unnamed: 31","Unnamed: 32","Unnamed: 33","Unnamed: 34","Unnamed: 35","Unnamed: 36","Unnamed: 37","Unnamed: 38","Unnamed: 39","Unnamed: 40","Unnamed: 41"],axis=1)

#document.isna().sum()

document.info()

#%%
lof_document = document[['Organization Name','Full Description']]

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
clf_lof = LocalOutlierFactor(n_neighbors=80,metric='euclidean',n_jobs=-1)
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


document["LOF Score"]= X_scores

document["LOF Score"][1]

print(type(document["LOF Score"]))
document=document.drop(["Organization Name","Full Description"],axis=1)




#%%
########################################################################
# calculate company age
########################################################################

document['Founded Date'] = pd.to_datetime(document['Founded Date'],errors="coerce")
document['Founded Date'] = pd.to_datetime(document['Founded Date']).dt.date
type(document['Founded Date'][0])
d = datetime.date(2020, 11, 5)
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

document=document.drop(['Founded Date'], axis=1)
document.isna().sum()
document.info()

document_fill = document.fillna(0)


###remove , in excel####
#%%
document_fill.to_csv(r"C:\Users\eric\Desktop\cb_ml\sepa_do.csv",index=False,encoding='utf8')

document_fill = pd.read_csv(r"C:\Users\eric\Desktop\cb_ml\sepa_do.csv",header=0,encoding="utf8")
#%%
document_fill["IPO Status"] = document_fill["IPO Status"].astype("category")
#%%

document_fill["Aberdeen - IT Spend"] = document_fill["Aberdeen - IT Spend"].astype(float)
document_fill["Aberdeen - IT Spend"].describe()

for i in range(len(document_fill)):
    if document_fill["Aberdeen - IT Spend"][i] >= 127363.096160:
        document_fill["Aberdeen - IT Spend"][i] = 1
    else:
        document_fill["Aberdeen - IT Spend"][i] = 0
#%%

document_fill["Estimated Revenue Range"] = document_fill["Estimated Revenue Range"].astype('category')
document_fill["Last Equity Funding Type"] = document_fill["Last Equity Funding Type"].astype('category')
document_fill["SEMrush - Visit Duration"] = document_fill["SEMrush - Visit Duration"].astype(float)
document_fill["SEMrush - Visit Duration"] = document_fill["SEMrush - Visit Duration"].astype(float)

document_fill=document_fill.drop(["Total Equity Funding Amount"],axis=1)
document_fill.info()
#%%


document_fill['Estimated Revenue Range'].replace(to_replace=['0', 'Less than $1M', '$1M to $10M','$10M to $50M','$50M to $100M','$100M to $500M','$500M to $1B','$1B to $10B','$10B+'], value=[1, 2, 3,4,5,6,7,8,9], inplace=True)

document_fill['Last Equity Funding Type'].replace(to_replace=['Angel', 'Corporate Round', 'Equity Crowdfunding','Initial Coin Offering','Post-IPO Equity','Pre-Seed','Private Equity','Seed','Series A','Series B','Series C','Series D','Series E','Series F','Series G','Series H','Undisclosed','Venture - Series Unknown'], value=[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], inplace=True)



document_fill["Estimated Revenue Range"] = document_fill["Estimated Revenue Range"].astype('category')
document_fill["Last Equity Funding Type"] = document_fill["Last Equity Funding Type"].astype('category')


document_fill = pd.concat([document_fill.drop('Estimated Revenue Range', axis=1), pd.get_dummies(document_fill['Estimated Revenue Range'],prefix="Estimated Revenue Range")], axis=1)
document_fill = pd.concat([document_fill.drop('Last Equity Funding Type', axis=1), pd.get_dummies(document_fill['Last Equity Funding Type'],prefix="Last Equity Funding Type")], axis=1)



document_fill.info()
document_fill.isna().sum()
#document_fill.to_csv(r"C:\Users\eric\Desktop\cb_ml\sepa_do1.csv",index=False,encoding='utf8')

#%%
####
X=document_fill.loc[:, document_fill.columns != 'IPO Status']
y=document_fill['IPO Status'] 
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

sm = SMOTE(random_state=2,k_neighbors=5)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


plt.hist(y_train_res)
plt.draw()

X_train_res.info()

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
#evaluation function
from scipy import interp

from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df
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
acc_logreg = round( metrics.accuracy_score(y_test, y_pred_LF) , 3)
print( 'Accuracy of Logistic Regression model : ', acc_logreg )

f1_logreg=f1_score(y_test, y_pred_LF, average='weighted')
roc_logreg=np.round(roc_auc_score(y_test, y_pred_LF, average='weighted'), decimals=4)
report_with_auc = class_report(
    y_true=y_test, 
    y_pred=logreg_2.predict(X_test), 
    y_score=logreg_2.predict_proba(X_test))

print(report_with_auc)
print('f1 score: ', f1_logreg)
print('auc  score: ', roc_logreg)
#%%



#%%
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
#%%
#tree visualization
from sklearn.tree import export_graphviz

#export_graphviz(clf, out_file='tree.dot')
export_graphviz(clf, out_file='tree_spurs_dt.dot', 
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
estimator = rf.estimators_[3]
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
for feat, importance in zip(X_train_res.columns, rf.feature_importances_):  
    temp = [feat, importance*100]
    featureImpList.append(temp)

fT_df = pd.DataFrame(featureImpList, columns = ['Feature', 'Importance'])
print (fT_df.sort_values('Importance', ascending = False))
#%%




#%%
svc = StandardScaler()
X_train_scaled = svc.fit_transform(X_train_res)
X_test_scaled = svc.transform(X_test)

clf_svc= svm.SVC()

params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(clf_svc,params,cv=5,n_jobs=-1)
grid.fit(X_train_scaled,y_train_res.ravel())

gird_svc=grid.best_estimator_
# Train the model using the training sets 
gird_svc.fit(X_train_scaled,y_train_res.ravel())

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
#%%




#%%
#ligthgbm
x_train = svc.fit_transform(X_train_res)
x_test = svc.transform(X_test)
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
mse_lgb = mean_squared_error(y_test, predict_test.round())
r2_lgb= r2_score(y_test, predict_test.round())
f1_lgb=f1_score(y_test, predict_test.round(), average='weighted')
roc_lgb=np.round(roc_auc_score(y_test, predict_test.round(), average='weighted'), decimals=4)
print('Mean squared error: ', mse_lgb)
print('R2 score: ', r2_lgb)


ax = lgb.plot_importance(lgb_model, max_num_features=14, figsize=(10,10))
plt.show()
#%%

import graphviz 
export_graphviz(estimator, out_file='tree_rf.dot', 
                feature_names = X.columns,
                class_names = ["0","1"],
                max_depth = 3, # 표현하고 싶은 최대 depth
                precision = 3, # 소수점 표기 자릿수
                filled = True, # class별 color 채우기
                rounded=True, # 박스의 모양을 둥글게
               )

dot_data = lgb.create_tree_digraph(lgb_model, tree_index = 1,show_info=['split_gain'])

graph = graphviz.Source(dot_data)  
graph 
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
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

search.fit(X_train_res, y_train_res.ravel())

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
#%%

#%%
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
            
grid.fit(X_train_res, y_train_res.ravel()) 

adaBoost_grid = grid.best_estimator_
# Train the model using the training sets 
adaBoost_grid.fit(X_train_res, y_train_res.ravel())

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
#%%
n_estimators = [100,140,145,150,160, 170,175,180,185];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=42)
learning_r = [0.1,1,0.01,0.5]

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
acc_ada = round( metrics.accuracy_score(y_pred_ada.round(), y_test) , 3 )
print( 'Accuracy of AdaBoost model : ', acc_ada )
mse_ada = mean_squared_error(y_test, y_pred_ada.round())
r2_ada= r2_score(y_test, y_pred_ada.round())
f1_ada=f1_score(y_test, y_pred_ada.round(), average='weighted')

#%%
report_with_auc = class_report(
    y_true=y_test, 
    y_pred=adaBoost_grid.predict(X_test), 
    y_score=adaBoost_grid.predict_proba(X_test))

print(report_with_auc)
print('Mean squared error: ', mse_ada)
print('R2 score: ', r2_ada)
#%%
#mlp classifier
# =============================================================================
# parameters_mlp = {'solver': ['lbfgs'], 'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
# =============================================================================

# =============================================================================
# clf_mlp = GridSearchCV(MLPClassifier(), parameters_mlp, n_jobs=-1)
# 
# 
# 
# clf_mlp.fit(X_train_res, y_train_res.ravel())
# 
# clf_mlp.best_params_
# =============================================================================

#%%
# Train the model using the training sets 

MLP = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

MLP.fit(X_train_res, y_train_res.ravel())

y_pred_mlp= MLP.predict(X_test)

accuracy_score(y_test, y_pred_mlp)
# Calculating the accuracy
acc_mlp = round( metrics.accuracy_score(y_pred_mlp.round(), y_test) , 3 )
print( 'Accuracy of MLP_Classifier : ', acc_mlp )

mse_mlp = mean_squared_error(y_test, y_pred_mlp.round())
r2_mlp= r2_score(y_test, y_pred_mlp)
f1_mlp=f1_score(y_test, y_pred_mlp, average='weighted')
roc_mlp=np.round(roc_auc_score(y_test, y_pred_mlp, average='weighted'), decimals=4)
print('Mean squared error: ', mse_mlp)
print('R2 score: ', r2_mlp)
#%%
print('Training set score: {:.4f}'.format(MLP.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(MLP.score(X_test, y_test)))
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_rf)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
#%%
plt.figure(figsize=(15, 4))
plt.plot(['Logistic','AdaBoost',"Decision Tree","RandonForest","Support Vector Machines","LightGBM","Xgboost"], [acc_logreg, acc_ada,acc_dt, acc_rf, acc_svm, acc_lgb,acc_xgb], 'ro')

plt.show()
#%%
