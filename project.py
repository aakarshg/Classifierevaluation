import pandas as pd 
import numpy as np
import string, re
import nltk
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer, CountVectorizer
from sklearn import naive_bayes,metrics, linear_model,svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sqlite3


stop_list = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
punctuation = list(string.punctuation)
stop_list = stop_list + punctuation +["rt", 'url']
nouns = ["gopdebate","donald","trump","hillary","clinton","Gary","Johnson","Jill","Stein"]

data = pd.read_csv("C:\\Users\\Adarsh\\Desktop\\alda\\Sentiment3wayequal.csv")

classifier =[]
def preprocess(tweet):
    if type(tweet)!=type(2.0):
        tweet = tweet.lower()
        tweet = " ".join(tweet.split('#'))
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+))','URL',tweet)
        tweet = re.sub("http\S+", "URL", tweet)
        tweet = re.sub("https\S+", "URL", tweet)
        tweet = re.sub('@[^\s]+','AT_USER',tweet)
        tweet = tweet.replace("AT_USER","")
        tweet = tweet.replace("URL","")
        tweet = tweet.replace(".","")
        tweet = tweet.replace('\"',"")
        tweet = tweet.replace('&amp',"")
        tweet = tweet.replace("gopdebate","")
        tweet = tweet.replace("donald","")
        tweet = tweet.replace("trump","")
        tweet = tweet.replace("hillary","")
        tweet = tweet.replace("clinton","")
        tweet = tweet.replace("gary","")
        tweet = tweet.replace("johnson","")
        tweet = tweet.replace("jill","")
        tweet = tweet.replace("stein","")
        tweet  = " ".join([word for word in tweet.split(" ") if word not in stop_list])
        tweet  = " ".join([word for word in tweet.split(" ") if re.search('^[a-z]+$', word)])
        tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split(" ")])
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = tweet.strip('\'"')
    else:
        tweet=''
    return tweet

data['processed_text'] = data.text.apply(preprocess)
categories = data.sentiment.unique()
categories  = categories.tolist()


x = data.processed_text.values
y = data.sentiment.values

categories = ['Positive','Negative' ,'Neutral']



x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.20, stratify = y)
vectorizer_bi = TfidfVectorizer(ngram_range=(1,2),
                             min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=False)
train_vectors_bi = vectorizer_bi.fit_transform(x_train)
test_vectors_bi = vectorizer_bi.transform(x_test)
total_bi = vectorizer_bi.fit_transform(x)

vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=False)
train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)
total = vectorizer.fit_transform(x)

##classifier_rbf = svm.SVC()
##classifier_rbf.fit(train_vectors, y_train)
##prediction_rbf = classifier_rbf.predict(test_vectors)

classifier_linear_bi = svm.SVC(kernel='linear')
classifier_linear_bi.fit(train_vectors_bi, y_train)
prediction_linear_bi = classifier_linear_bi.predict(test_vectors_bi)

classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, y_train)
prediction_linear = classifier_linear.predict(test_vectors)

scores= cross_val_score(classifier_linear,total,y, cv=10)
##mcc = matthews_corrcoef(prediction_linear, y_test)


print "......unigram svm......."
print "cross validation scores for unigram svm" , scores
##print "The Matthews correlation coefficient for unigram svm" , mcc
print "Accuracy of unigram SVM",accuracy_score(y_test, prediction_linear)
print "Confusion matrix of unigram SVM",confusion_matrix(y_test, prediction_linear)

scores_bi= cross_val_score(classifier_linear_bi,total_bi,y, cv=10)
##mcc_bi = matthews_corrcoef(prediction_linear_bi, y_test)

print "...... bigram svm......."
print "cross validation scores for bigram svm" , scores_bi
##print "The Matthews correlation coefficient for biigram svm" , mcc_bi
print "Accuracy of bigram SVM",accuracy_score(y_test, prediction_linear_bi)
print "Confusion matrix of bigram SVM",confusion_matrix(y_test, prediction_linear_bi)


print "......knn......."
neigh2 = KNeighborsClassifier(n_neighbors=2)
neigh3 = KNeighborsClassifier(n_neighbors=3)
neigh4 = KNeighborsClassifier(n_neighbors=4)
neigh5 = KNeighborsClassifier(n_neighbors=5)
neigh6 = KNeighborsClassifier(n_neighbors=6)
neigh7= KNeighborsClassifier(n_neighbors=7)
neigh8= KNeighborsClassifier(n_neighbors=8)
neigh9= KNeighborsClassifier(n_neighbors=9)
neigh10= KNeighborsClassifier(n_neighbors=10)

neigh2.fit(train_vectors_bi, y_train)
neigh3.fit(train_vectors_bi, y_train)
neigh4.fit(train_vectors_bi, y_train)
neigh5.fit(train_vectors_bi, y_train)
neigh6.fit(train_vectors_bi, y_train)
neigh7.fit(train_vectors_bi, y_train)
neigh8.fit(train_vectors_bi, y_train)
neigh9.fit(train_vectors_bi, y_train)
neigh10.fit(train_vectors_bi, y_train)

knn2 = neigh2.predict(test_vectors_bi)
knn3 = neigh3.predict(test_vectors_bi)
knn4 = neigh4.predict(test_vectors_bi)
knn5 = neigh5.predict(test_vectors_bi)
knn6 = neigh6.predict(test_vectors_bi)
knn7 = neigh7.predict(test_vectors_bi)
knn8 = neigh8.predict(test_vectors_bi)
knn9 = neigh9.predict(test_vectors_bi)
knn10 = neigh10.predict(test_vectors_bi)

scores = cross_val_score(neigh5,total_bi,y, cv=10)
##mcc_knn2= matthews_corrcoef(knn2, y_test)
##mcc_knn3= matthews_corrcoef(knn3, y_test)
##mcc_knn4= matthews_corrcoef(knn4, y_test)
##mcc_knn5= matthews_corrcoef(knn5, y_test)
##mcc_knn6= matthews_corrcoef(knn6, y_test)
##mcc_knn7= matthews_corrcoef(knn7, y_test)
##mcc_knn8= matthews_corrcoef(knn8, y_test)
##mcc_knn9= matthews_corrcoef(knn9, y_test)
##mcc_knn10= matthews_corrcoef(knn10, y_test)
##print "The Matthews correlation coefficient for bigram KNN" , mcc_knn2,mcc_knn3,mcc_knn4,"mcc for neighbours = 5 :",mcc_knn5,mcc_knn6,mcc_knn7,mcc_knn8,mcc_knn9,mcc_knn10
print "cross validation scores for bigram knn of 5 neighbours",scores
print "Accuracy of bigram KNN", accuracy_score(y_test, knn2),accuracy_score(y_test, knn3),accuracy_score(y_test, knn4),"accuracy for neighbours = 5 :",accuracy_score(y_test, knn5),accuracy_score(y_test, knn6),accuracy_score(y_test, knn7),accuracy_score(y_test, knn8),accuracy_score(y_test, knn9),accuracy_score(y_test, knn10)
print "Confusion matrix of bigram KNN of 5 neighbours",confusion_matrix(y_test, knn5)




