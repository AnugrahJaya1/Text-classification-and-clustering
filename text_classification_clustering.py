#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:36:07 2018
@author: abhijeet
"""

# Importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter

from sklearn.model_selection import KFold
from sklearn import metrics

#import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')


stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

# Cleaning the text sentences so that punctuation marks, stop words & digits are removed  
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y


print ("\nThere are 10 sentences of following three classes on which K-NN classification and K-means clustering"\
         " is performed : \n1. Cricket \n2. Artificial Intelligence \n3. Chemistry")
path = "Sentences.txt"

train_clean_sentences = []
fp = open(path,'r')
for line in fp:
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)
       
vectorizer = TfidfVectorizer(stop_words='english')
# data x
X = vectorizer.fit_transform(train_clean_sentences)

# Creating true labels for 30 training sentences 
# data y
y_train = np.zeros(30)
y_train[10:20] = 1
y_train[20:30] = 2


true_test_labels = ['Cricket','AI','Chemistry']

#Data set untuk test
test_sentences = ["Chemical compunds are used for preparing bombs based on some reactions",\
                  "Cricket is a boring game where the batsman only enjoys the game",\
                  "Machine learning is an area of Artificial intelligence"]

# membersihkan data
test_clean_sentence = []
for test in test_sentences:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+","",cleaned)
    test_clean_sentence.append(cleaned)

# Data set yang siap digunakan untuk test
Test = vectorizer.transform(test_clean_sentence) 

#print data set untuk test
print ("\nBelow 3 sentences will be predicted against the learned nieghbourhood and learned clusters:\n1. ",\
        test_sentences[0],"\n2. ",test_sentences[1],"\n3. ",test_sentences[2])

print("\n-------------------PREDICTION USING CLASSIFICATION-------------------")

#Classification

#KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

modelknn = KNeighborsClassifier(n_neighbors=5)
modelknn.fit(X,y_train)
predicted_labels_knn = modelknn.predict(Test)

print ("\n-------------------------PREDICTIONS BY KNN-------------------------")
print ("\n",test_sentences[0],":",true_test_labels[np.int(predicted_labels_knn[0])],\
        "\n",test_sentences[1],":",true_test_labels[np.int(predicted_labels_knn[1])],\
        "\n",test_sentences[2],":",true_test_labels[np.int(predicted_labels_knn[2])],"\n")

# Data untuk split ke dalam KFold
X = X.toarray()
Y = y_train

kf = KFold(n_splits=10)
precision=[]
recall=[]
for train_id, test_id in kf.split(X):
    x_train, x_test = X[train_id], X[test_id]
    y_train, y_test = Y[train_id], Y[test_id]
    modelknn.fit(x_train, y_train);
    y_pred = modelknn.predict(x_test)
    precision.append(precision_score(y_test, y_pred,average='macro'))
    recall.append(recall_score(y_test, y_pred,average='macro'))
    

print("Precision dengan knn","dengan k =",5," = ",np.mean(precision))
print("Recall dengan knn dengan k=",5,"=",np.mean(recall),"\n")

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb.fit(X,Y)
predicted_labels_nb = nb.predict(Test.toarray())

print ("\n-------------------------PREDICTIONS BY NAIVE BAYES-------------------------")
print ("\n",test_sentences[0],":",true_test_labels[np.int(predicted_labels_nb[0])],\
        "\n",test_sentences[1],":",true_test_labels[np.int(predicted_labels_nb[1])],\
        "\n",test_sentences[2],":",true_test_labels[np.int(predicted_labels_nb[2])],"\n")

precision=[]
recall=[]
for train_id, test_id in kf.split(X):
    x_train, x_test = X[train_id], X[test_id]
    y_train, y_test = Y[train_id], Y[test_id]
    nb.fit(x_train, y_train);
    y_pred = nb.predict(x_test)
    precision.append(precision_score(y_test, y_pred,average='macro'))
    recall.append(recall_score(y_test, y_pred,average='macro'))
    
print("Precision dengan naive bayes = ",np.mean(precision),)
print("Recall dengan naive bayes=",np.mean(recall),"\n")

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

tree.fit(X,Y)
predicted_labels_tree=tree.predict(Test)

print ("\n-------------------------PREDICTIONS BY Decesion Tree-------------------------")
print ("\n",test_sentences[0],":",true_test_labels[np.int(predicted_labels_tree[0])],\
        "\n",test_sentences[1],":",true_test_labels[np.int(predicted_labels_tree[1])],\
        "\n",test_sentences[2],":",true_test_labels[np.int(predicted_labels_tree[2])])

precision=[]
recall=[]
for train_id, test_id in kf.split(X):
    x_train, x_test = X[train_id], X[test_id]
    y_train, y_test = Y[train_id], Y[test_id]
    tree.fit(x_train, y_train);
    y_pred = tree.predict(x_test)
    precision.append(precision_score(y_test, y_pred,average='macro'))
    recall.append(recall_score(y_test, y_pred,average='macro'))
    
print("\nPrecision dengan decision tree = ",np.mean(precision),)
print("Recall dengan decision tree=",np.mean(recall),"\n")


# Clustering the training 30 sentences with K-means technique
from sklearn.cluster import KMeans
modelkmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=100)
modelkmeans.fit(X)
predicted_labels_kmeans = modelkmeans.predict(Test)

print('--------------------------- Clustering ---------------------------')

precision=[]
recall=[]
dist = []
for train_id, test_id in kf.split(X):
    x_train, x_test = X[train_id], X[test_id]
    y_train, y_test = Y[train_id], Y[test_id]
    modelkmeans.fit(x_train);
    y_pred = modelkmeans.predict(x_test)
    precision.append(precision_score(y_test, y_pred,average='macro'))
    recall.append(recall_score(y_test, y_pred,average='macro'))
    
print("\nPrecision dengan K-means = ",np.mean(precision),)
print("Recall dengan K-means=",np.mean(recall),"\n")

#Clustering menggunakan agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

modelAgglo=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
modelAgglo.fit(X)
#predicted_labels_kmeans=modelAgglo.fit_predict(Test)
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

precision=[]
recall=[]
for train_id, test_id in kf.split(X):
    x_train, x_test = X[train_id], X[test_id]
    y_train, y_test = Y[train_id], Y[test_id]
    modelAgglo.fit(x_train);
    y_pred = modelAgglo.fit_predict(x_test)
    precision.append(precision_score(y_test, y_pred,average='macro'))
    recall.append(recall_score(y_test, y_pred,average='macro'))
    
print("\nPrecision dengan Agglomerative Clustering = ",np.mean(precision),)
print("Recall dengan Agglomerative Clustering=",np.mean(recall),"\n")