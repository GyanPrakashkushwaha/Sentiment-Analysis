from keras.models import load_model
import os
# print(os.getcwd())
# os.chdir(r'd:\\vscode_machineLearning\\internship\\sentiment-Analysis-fellowship.ai')
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np


class preprocess_input:
    def __init__(self, text):
        self.text = text.lower()  
    
    # remove html tags
    def remove_html_tags(self):
        soup = BeautifulSoup(self.text, "html.parser")
        return soup.get_text()
    
    def remove_between_square_brackets(self):
        return re.sub(r'http\S+', '', self.text)
    
    def remove_stopwords(self):
        stop = set(stopwords.words('english'))
        final_text = []
        for i in self.text.split():
            if i.strip().lower() not in stop and i.strip().lower().isalpha():
                final_text.append(i.strip().lower())
        return " ".join(final_text)
    
    def use_all(self):
        txt = self.remove_html_tags()
        txt = self.remove_between_square_brackets()
        txt = self.remove_stopwords()
        return txt
    
    def __str__(self) -> str:
        return self.use_all()
    


def tokenize_vectorize_input(text:str):
    ps = PorterStemmer()
    stemmed_words =[ps.stem(word) for word in str(text).split()]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=stemmed_words)
    docs = tokenizer.texts_to_sequences([stemmed_words])
    pad_docs = pad_sequences(sequences=docs,maxlen=150,padding='post')

    return np.array(pad_docs)
    
# txt_to_predict=tokenize_vectorize_input(obj.text)


# obj = preprocess_input(text=txt)
# print(obj)
