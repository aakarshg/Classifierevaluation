import sqlite3
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas as pd
import sklearn
import cPickle
import numpy as np
import HTMLParser as ht
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import random
from random import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef

def split_into_lemmas(message):
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

def preproc(tweet):
    if type(tweet)!=type(2.0):
        cleaned="".join(re.findall('[A-Z][^A-Z]*',tweet))
        cleaned =  re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', cleaned)
        cleaned=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",cleaned).split())
        cleaned=re.sub('RT ', '', cleaned)
        tweet=cleaned
    else:
        tweet=''
    return tweet

i=1
pos_l=[]#positive list
neg_l=[]#negative list
neu_l=[]#neutral list
full_l=[]#full list

sql_conn = sqlite3.connect('database.sqlite')
messages=pd.read_sql("SELECT text,sentiment from Sentiment", sql_conn)
for index, row in messages.iterrows():
    h_p=ht.HTMLParser()
    tweet=row['text']
    cleaned=preproc(tweet)
    row['text']=cleaned
    if row['sentiment']=='Positive':
        full_l.append([row['text'],row['sentiment']])
        pos_l.append(row['text'])
    if row['sentiment']=='Negative':
        full_l.append([row['text'],row['sentiment']]) 
        neg_l.append(row['text'])
    if row['sentiment']=='Neutral':
        #next line to include neutral samples
        full_l.append([row['text'],row['sentiment']]) 
        neu_l.append(row['text'])
        

#shuffle(full_l)#To ensure a full shuffle happens
#SAMPLING BEGINS
#'''
ran_posl=random.sample(pos_l,2000)
ran_negl=random.sample(neg_l,8493)
ran_neul=random.sample(neu_l,2265)

neulsent=[]
poslsent=[]
neglsent=[]

nsent=[]
psent=[]
nusent=[]
for i in xrange(8493):
    nsent.append('Negative')
for i in xrange(2236):
    psent.append('Positive')
for i in xrange(3142):
    nusent.append('Neutral')

#total=8493+2236
#total=10729
#30% of tot=3219 

#'''
for i in xrange(2000):
    neulsent.append('Neutral')
    neglsent.append('Negative')
    poslsent.append('Positive')

#'''
#we are combining list as positive then neutral and then negative
sentl=poslsent+neglsent+neulsent
textl=ran_posl+ran_negl+ran_neul
'''
#POS:324,1622
#NEG:1224,4898
#NEU:453,2265

#if we wanted equal sampling

sentl_test=poslsent[1600:]+neglsent[1600:]+neulsent[1600:]
textl_test=ran_posl[1600:2000]+ran_negl[1600:2000]+ran_neul[1600:2000]
sentl_train=poslsent[:1600]+neglsent[:1600]+neulsent[:1600]
textl_train=ran_posl[:1600]+ran_negl[:1600]+ran_neul[:1600]

#'''

#'''

#Strat 3 way

sentl_train=psent[322:1612]+nsent[1224:6123]+nusent[453:2265]
textl_train=ran_posl[322:1612]+ran_negl[1224:6123]+ran_neul[453:2265]
sentl_test=psent[:322]+nsent[:1224]+nusent[:453]
textl_test=ran_posl[:322]+ran_negl[:1224]+ran_neul[:453]

#testing against whole of negative + positive population:
text_t=pos_l+neg_l#+neu_l
sent_t=psent+nsent#+nusent
#'''
'''
#for random sampling
#random sampling more like shuffling:
#full_l=random.sample(full_l,10729)#change the second number to the samples u want to take
full_l=random.sample(full_l,13871)#including neutral
sentl_train=[]
textl_train=[]
sentl_test=[]
textl_test=[]
for i in range(2774):#change the stop values according to the sample size and ratio :)
    textl_test.append(full_l[i][0])
    sentl_test.append(full_l[i][1])

for i in range(2774,13871):
    textl_train.append(full_l[i][0])
    sentl_train.append(full_l[i][1])

#'''

#Stratified Sampling
'''
sentl_train=psent[:1600]+nsent[:6400]
textl_train=ran_posl[:1600]+ran_negl[:6400]

sentl_test=psent[1600:2000]+nsent[6400:8000]
textl_test=ran_posl[1600:2000]+ran_negl[6400:8000]
#'''


bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(textl_train)

#print len(bow_transformer.vocabulary_)
#message4=messages['text'][3]
#print message4
#bow4 = bow_transformer.transform([message4])
bowt=bow_transformer.transform(textl_test)
#print bow4
#print bow4.shape
messages_bow = bow_transformer.transform(textl_train)
#print 'sparse matrix shape:', messages_bow.shape
tfidf_transformer = TfidfTransformer().fit(messages_bow)
#tfidf4 = tfidf_transformer.transform(bow4)
tfid_test=tfidf_transformer.transform(bowt)
#print tfidf4
messages_tfidf = tfidf_transformer.transform(messages_bow)
#print 'The shape of TF-IDF'
#print messages_tfidf.shape
nbclass = MultinomialNB().fit(messages_tfidf, sentl_train)
#print 'predicted:', nbclass.predict(tfidf4)[0]
#print 'expected:', messages.sentiment[3]

all_predictions = nbclass.predict(tfid_test)


print "For stratified sampling 3way"
print 'accuracy', accuracy_score(sentl_test, all_predictions)
print 'confusion matrix\n', confusion_matrix(sentl_test, all_predictions)
print '(row=expected, col=predicted)'

#print textl_test.shape
#print "sentiment shape"
#print sentl_test.shape
'''
scores=cross_val_score(nbclass, tfid_test, sentl_test,cv=9,scoring='f1_macro')
print "F1 scores are",scores
f1score=f1_score(sentl_test,all_predictions,average='weighted')
print "weighted f-1 score is",f1score
'''

f1score=f1_score(sentl_test,all_predictions,average='macro')
print "macro f-1 score is",f1score


'''
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores=cross_val_score(nbclass, tfid_test, sentl_test,cv=9)
print "normal scores are",scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
mscores=matthews_corrcoef(sentl_test, all_predictions)
print "Mathews corrcoef is",mscores

with open('nb_ug.pkl', 'wb') as fout:
    cPickle.dump(nbclass, fout)
'''
