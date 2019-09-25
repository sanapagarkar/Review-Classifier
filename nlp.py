import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #3 is for ignoring double quotes "

#cleaning text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] #list of cleaned reviews
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #cleaned review in variable review, following ^ are the things that you dont want to remove
    review =  review.lower() #makes all letters lower case
    review = review.split()
    ps = PorterStemmer()
    #review = [word for word in review if not word in stopwords.words('english')] # put set(stopwords.words('english')) for larger text say article etc bc python works faster with sets as compared to lists
    #review = [ps.stem(word) for word in review]
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] 
    review = ' '.join(review)
    corpus.append(review)
    
#Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)#max_features discards the least frequently used words #can clean text automatically by passing suitable arguments but doing it manually as we did gives more space to do things the way we want
x= cv.fit_transform(corpus).toarray() #independent variable
y= dataset.iloc[:,1].values #dependent variable

#classification model (Naive Bayes)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#create classifier here
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (55+91)/200 #test set info from cm