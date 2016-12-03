import sqlite3
import pandas as pd
import numpy as np
import HTMLParser as ht
import re
from nltk import NaiveBayesClassifier as nbc
from nltk.tokenize import word_tokenize
from itertools import chain
import random
import nltk as nltk


i=1
pos_l=[]#positive list
neg_l=[]#negative list
neu_l=[]#neutral list

sql_conn = sqlite3.connect('database.sqlite')
df=pd.read_sql("SELECT text,sentiment from Sentiment", sql_conn)
#for index, row in x.iterrows():
 #   if i<22:
  #      print row['text']
   #     i=i+1
#mydict=dict(zip(list(df.text),list(df.sentiment)))
#first2pairs = {k: mydict[k] for k in mydict.keys()[:5]}
#training_data= first2pairs.items()
elements = []
for index, row in df.iterrows():
    h_p=ht.HTMLParser()
    tweet=row['text']
    cleaned="".join(re.findall('[A-Z][^A-Z]*',tweet))
    cleaned =  re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', cleaned)
    cleaned=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",cleaned).split())
    cleaned=re.sub('RT ', '', cleaned)
    row['text']=cleaned
    if row['sentiment']=='Positive':
        pos_l.append(row['text'])
    if row['sentiment']=='Negative':
        neg_l.append(row['text'])
    if row['sentiment']=='Neutral':
        neu_l.append(row['text'])    

print "Size of neutral list",len(neu_l)
print "Size of positive list",len(pos_l)
print "Size of negative list",len(neg_l)

ran_posl=random.sample(pos_l,1000)
ran_negl=random.sample(neg_l,1000)
ran_neul=random.sample(neu_l,1000)


neulsent=[]
poslsent=[]
neglsent=[]

for i in xrange(1000):
    neulsent.append('Neutral')
    neglsent.append('Negative')
    poslsent.append('Positive')
    
#we are combining list as positive then neutral and then negative

sentl=poslsent+neulsent+neglsent
textl=ran_posl+ran_neul+ran_negl


#if we wanted equal sampling
sentl_train=poslsent[:700]+neulsent[:700]+neglsent[:700]
textl_train=ran_posl[:700]+ran_negl[:700]+ran_neul[:700]
sentl_test=poslsent[700:]+neulsent[700:]+neglsent[700:]
textl_test=ran_posl[700:]+ran_negl[700:]+ran_neul[700:]



ts=sentl
tl=textl

#tl=list(df.text)
#ts=list(df.sentiment)


tl=tl[:2]#sample it using random
ts=ts[:2]
#print ts
#feat_set=dict(feat_set)

training_data=zip(tl,ts)
#training_data=dict(training_data)


#training_data, test_set = feat_set[:700],feat_set[700:]

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))

feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in training_data]

classifier = nbc.train(feature_set)

#for classifying a new sentence

test_sentence = tl[1]
featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary}

print "test_sent:",test_sentence
print "tag:",classifier.classify(featurized_test_sentence)

#print nltk.classify.accuracy(classifier,test_set)


