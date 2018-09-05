import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

class LDA:

    def __init__(self, data):
         
        labels = ["text", "index"]
        data = pd.DataFrame.from_records(data, columns=labels)
        data_text = data[["text"]]
        data_text["index"] = data_text.index
        self.documents = data_text
        self.stemmer = SnowballStemmer('english')
    
    def get_docs(self):
        return self.documents["text"].map(self.__preprocess)

    def __preprocess(self, text):
        """
            Tokenizes given text into sentences and words. 
            Lowercase and punctuation removed.
            Words smaller than 3 characters removed.
            Stopwords removed.
            Lemmatize words.
            Stem words.
        """
        result = []
        for token in simple_preprocess(text):
            if token not in STOPWORDS and len(token) > 3:
                result.append(self.__lemmatize_stemming(token))
        
        return result

    def __lemmatize_stemming(self, text):
        """
            Lemmatizes and stems english words
        """
        return self.stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v')) 
 