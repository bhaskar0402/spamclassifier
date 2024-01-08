import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import re

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

    #stop_word=set(stopwords.words('english'))
    #print(text)
    #review=re.sub('[^a-zA-Z0-9]',' ',text)
    #review=review.lower()
    #review=review.split()
    #review = [ps.stem(word) for word in review if not word in stop_word]
    #review = ' '.join(review)

    #return review

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    print('1. preprocess')
    transformed_sms = transform_text(input_sms)
    print(transformed_sms)
    print('2. vectorize')
    vector_input = tfidf.transform([transformed_sms])
    print('3. predict')
    result = model.predict(vector_input)[0]
    print('4. Display')
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")