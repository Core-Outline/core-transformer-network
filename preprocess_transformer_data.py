import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, Input, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras import Model, Input
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequence

import re
import string

class Preparation:
    """
    Removing punctuations like . , ! $( ) * % @
    Removing URLs
    Removing Stop words
    Lower casing
    Tokenization
    Stemming
    Lemmatization
    TextVectorization
    """
    def __init__(self, max_features, text, embedding_dim, max_length):
        self.text = text
        self.MAX_FEATURES = max_features
        self.EMBEDDING_DIM = embedding_dim
        self.MAX_LENGTH  = max_length
    
    def remove_punctuation(self):
        return "".join([i for i in self.text if i not in string.punctuation])
    
    def lower_case(self):
        return self.text.lower()
    
    def word_tokenize_(self):
        return nltk.word_tokenize(self.text)
    
    def sent_tokenize_(self):
        START_TOKEN = "+==="
        END_TOKEN = "===+"
        result = [(START_TOKEN + i + END_TOKEN) for i in nltk.sent_tokenize(self.text) if len(START_TOKEN + i + END_TOKEN) < self.MAX_LENGTH] 
        return result
    
    def remove_stopwords(self, tokens):
        stopwords = nltk.corpus.stopwords.words('english')
        return [token for token in tokens if token not in stopwords]
    
    def stem_words(self, tokens):
        porter_stemmer = PorterStemmer()
        return [porter_stemmer.stem(token) for token in tokens]
    
    def lemmatize_words(self, tokens):
        wordnet_lemmatizer = WordNetLemmatizer()
        return [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    
    def remove_urls(self, text):
        return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    
    def text_vectorization(self, text_arr):
        vectorize_layer = TextVectorization(
            max_tokens = self.MAX_FEATURES,
            output_mode = 'int',
            output_sequence_length = 500
        )
        vectorize_layer.adapt(text_arr, batch_size=64)
        
        X_train_padded =  vectorize_layer(text_arr)
        X_train_padded = X_train_padded.np()
        X_train_padded = np.reshape(X_train_padded, (np.shape(X_train_padded)[0], np.shape(X_train_padded)[1], 1 ))

        return X_train_padded
    
    def word_map_tokenizer(self, text_arr):
        tokenizer = Tokenizer(num_words = 100, oov_token = '<00V>')
        self.tokenizer = tokenizer.fit_on_texts(text_arr)

    def get_id_seq(self, text):
        return self.tokenizer.texts_to_sequences([text])
    
    def get_text_seq(self, id_seq):
        return self.tokenizer.sequences_to_text([id_seq])
    
    def pad_seq_(self, )
    




    
    


