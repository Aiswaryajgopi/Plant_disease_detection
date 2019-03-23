import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn import metrics
#from sklearn import svm
image=pd.read_csv('train_c.csv')
print(image.head())
x_train,x_test,y_train,y_test=train_test_split(image.iloc[0],random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#clf = MultinomialNB().fit(X_train_tfidf, y_train)
clf=svm.SVC(kernel='nonlinear')
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print ("accuracy: ",metrics.accuracy_score(y_test,y_pred))

