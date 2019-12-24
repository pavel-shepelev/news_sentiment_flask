from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import json
import time
import nltk
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
"""
@app.route('/preprocess',methods=['POST'])
def preprocess():
    return: 0
"""
@app.route('/preprocess',methods=['GET', 'POST'])
def preprocess():
    if request.method == 'POST':
        message = request.form['message']
        
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction, text=message.capitalize())


@app.route('/predict',methods=['POST'])
def predict():
    start = time.time()
    """
    with open('train.json') as f:
        raw_train = json.load(f)
    with open('test.json') as f:
        raw_test = json.load(f)
    df= pd.DataFrame.from_dict(raw_train)
    
    X = df['text']
    y = df['sentiment']
       
    stemmer = RussianStemmer()
    stemmed = []
    Xp =[]
    for i in range(len(X)):
        for word in X[i].split(' '): 
            s= stemmer.stem(word)
            if s not in stopwords.words('russian'):
                stemmed.append(s)
                #proc = [x for x in stemmed if x not in stopwords.words('russian')]
                st =' '.join(word for word in stemmed)
                st =st.translate(str.maketrans('', '', string.punctuation))
        Xp.append(st)
        stemmed = []
        st=''
        print(i)
    Xp =pd.Series(Xp).astype(str)

    tfidf = TfidfVectorizer()
    Xp = tfidf.fit_transform(Xp)
    X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.33, random_state=10)
    
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    
    """    
    score = open('tfidf.pkl','rb')
    tfidf = joblib.load(score)
    
    model = open('model2.pkl','rb')
    clf = joblib.load(model)
    
    if request.method == 'POST':
        message = request.form['message']
        
        stemmer = RussianStemmer()
        stemmed = []
        for word in message.split(' '):
            stemmed.append(stemmer.stem(word))
        proc = [x for x in stemmed if x not in stopwords.words('russian')]  
        st =' '.join(word for word in proc)
        st= st.translate(str.maketrans('', '', string.punctuation))
        
        data = [st]
        vect = tfidf.transform(data).toarray()
        my_prediction = clf.predict(vect)
        
        feature_array = np.array(tfidf.get_feature_names())
        tfidf_sorting = np.argsort(vect).flatten()[::-1]

        n = 5
        top_n = feature_array[tfidf_sorting][:n]
        
        end = time.time()
        final_time = end-start
        
        
    return render_template('result.html',prediction = my_prediction[0], text=message, stemmed=st, final_time=final_time, top = top_n)



if __name__ == '__main__':
	app.run(debug=True)