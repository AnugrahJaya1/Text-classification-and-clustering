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
acc=[]
for train_id, test_id in kf.split(X):
    x_train, x_test = X[train_id], X[test_id]
    y_train, y_test = Y[train_id], Y[test_id]
    modelknn.fit(x_train, y_train);
    y_pred = modelknn.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, y_pred))

print("Akurasi dengan knn","dengan k =",5," = ",np.mean(acc),"\n")

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb.fit(X,Y)
predicted_labels_nb = nb.predict(Test.toarray())

print ("\n-------------------------PREDICTIONS BY NAIVE BAYES-------------------------")
print ("\n",test_sentences[0],":",true_test_labels[np.int(predicted_labels_nb[0])],\
        "\n",test_sentences[1],":",true_test_labels[np.int(predicted_labels_nb[1])],\
        "\n",test_sentences[2],":",true_test_labels[np.int(predicted_labels_nb[2])],"\n")

acc=[]
for train_id, test_id in kf.split(X):
    x_train, x_test = X[train_id], X[test_id]
    y_train, y_test = Y[train_id], Y[test_id]
    nb.fit(x_train, y_train);
    y_pred = nb.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, y_pred))
    
print("Akurasi dengan naive bayes = ",np.mean(acc),"\n")

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

tree.fit(X,Y)
predicted_labels_tree=tree.predict(Test)

print ("\n-------------------------PREDICTIONS BY Decesion Tree-------------------------")
print ("\n",test_sentences[0],":",true_test_labels[np.int(predicted_labels_tree[0])],\
        "\n",test_sentences[1],":",true_test_labels[np.int(predicted_labels_tree[1])],\
        "\n",test_sentences[2],":",true_test_labels[np.int(predicted_labels_tree[2])])

acc=[]
for train_id, test_id in kf.split(X):
    x_train, x_test = X[train_id], X[test_id]
    y_train, y_test = Y[train_id], Y[test_id]
    tree.fit(x_train, y_train);
    y_pred = tree.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, y_pred))
    
print("Akurasi dengan decision tree = ",np.mean(acc),"\n")


# Clustering the training 30 sentences with K-means technique
from sklearn.cluster import KMeans
modelkmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=100)
modelkmeans.fit(X)
predicted_labels_kmeans = modelkmeans.predict(Test)

print ("\n-------------------------PREDICTIONS BY K-Means-------------------------")
print ("\nIndex of Cricket cluster : ",Counter(modelkmeans.labels_[0:10]).most_common(1)[0][0])
print ("Index of Artificial Intelligence cluster : ",Counter(modelkmeans.labels_[10:20]).most_common(1)[0][0]) 
print ("Index of Chemistry cluster : ",Counter(modelkmeans.labels_[20:30]).most_common(1)[0][0])

print ("\n",test_sentences[0],":",true_test_labels[np.int(predicted_labels_kmeans[0])],\
        "\n",test_sentences[1],":",true_test_labels[np.int(predicted_labels_kmeans[1])],\
        "\n",test_sentences[2],":",true_test_labels[np.int(predicted_labels_kmeans[2])],"\n")

acc = []
dist = []
for train_id, test_id in kf.split(X):
    x_train, x_test = X[train_id], X[test_id]
    y_train, y_test = Y[train_id], Y[test_id]
    modelkmeans.fit(x_train);
    y_pred = modelkmeans.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, y_pred))
    
print("Akurasi dengan K-Means = ",np.mean(acc),"\n")

