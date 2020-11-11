# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:51:47 2020

@author: eric
"""

import pandas as pd
import numpy as np
import codecs
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split

with codecs.open("cb_healthcare.csv", "r",encoding='utf-8', errors='ignore') as file_data:
     cb_healthcare = pd.read_csv(file_data)

cb_healthcare.head()
cb_healthcare=cb_healthcare.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4","Unnamed: 5","Unnamed: 6","Unnamed: 7","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12"],axis=1)

cb_healthcare.head()

len(cb_healthcare)

cb_healthcare = cb_healthcare.dropna()

cb_healthcare.head()

#%%
cb_healthcare=cb_healthcare.rename(columns={'Full Description': 'Text'})


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
    model = Doc2Vec(alpha=0.025, min_alpha=0.025)
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

#X,y=vector_list[0:1200],vector_list[1201:1226]

df_x = pd.DataFrame(X)
clf_lof = LocalOutlierFactor(n_neighbors=20,metric='euclidean')
y_pred = clf_lof.fit_predict(df_x)


X_scores = -(clf_lof.negative_outlier_factor_)

#%%
clf = LocalOutlierFactor(n_neighbors=80,metric='euclidean')
y_pred = clf.fit_predict(df_x)
y_pred_score = clf._decision_function(df_x)
scores_doc = -(clf.negative_outlier_factor_)

lofScores_10 = pd.DataFrame(scores_doc,columns = ['LOF_scores'])
#lofScores_10['N/M']=patent['기업이름']
sort = lofScores_10.sort_values(["LOF_scores"],ascending=[False])
sorts2 = np.array(sort)
#%%
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,5))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
plt.title("Local Outlier Factor (LOF)",fontsize = 20)
#plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
b = plt.scatter(X_embedded[61:, 0], X_embedded[61:, 1], c='blue',
                edgecolor='k', s=20)
a = plt.scatter(X_embedded[:60,0 ], X_embedded[:60, 1], c='red',
                edgecolor='k', s=30)

#plt.axis('tight')
#plt.xlim((-1, 2))
#plt.ylim((-2, 2))
plt.legend([a, b],
           ["abnormal observations",
            "normal observations"],
           loc="upper left")
plt.show()
#%%
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X)
 X_embedded[:, 0]
