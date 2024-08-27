Financial Market News - Sentiment Analysis
this is a data of Financial Market Top 25 News for the day and task is to Train and Predict Models for overall sentiment Analysis
import Library
import pandas as pd
import numpy as np
import dataset
df = pd.read_csv(r"https://raw.githubusercontent.com/YBI-Foundation/Dataset/Main/Financial%20Market%20News.csv", encoding = 'ISO-8859_1')
df.head()
df.info()
df.shape
df.columns
                                                                                                  
Get Feature Selection
' '. join(str(x) for x in df.iloc[1,2:27])
df.index
len(df.index)
news = []
type(news)
news[0]
x = news
type(x)
     
Get Feature Text Conversion to Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(lowercase = True, ngram_range=(1,1))
x = cv.fit_transform(x)
x.shape
y = df['label']
y.shape
     
Get Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train _test_split(x,y,test_size = 0.3, stratify = y, random_state = 2529)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
from sklearn.metrics import Classification report , confusion matrix, accuracy score
confusion_matric(y_test, y_pred)
print(Clasification_report(y_test, y_pred)
