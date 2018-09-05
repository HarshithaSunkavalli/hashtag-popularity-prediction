import pandas as pd
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

class LDA:

    __MINIMUM_DOCUMENT_APPEARANCES = 0
    __MAXIMUM_DOCUMENT_APPEARANCE_FRACTION = 1
    __TOKENS_TO_KEEP = 100000

    def __init__(self, data):
         
        labels = ["text", "index"]
        data = pd.DataFrame.from_records(data, columns=labels)
        data_text = data[["text"]]
        data_text["index"] = data_text.index
        self.__documents = data_text
        self.__stemmer = SnowballStemmer('english')
        self.__processed_docs = self.__documents["text"].map(self.__preprocess)
        self.__bow_corpus = self.__filter_out_words()
        self.lda_with_bag_of_words()

    def lda_with_bag_of_words(self):
        """
            lda using bag of words
        """

        lda_model = models.LdaMulticore(self.__bow_corpus, num_topics=20, id2word=self.__dictionary, passes=2, workers=2)

        #for idx, topic in lda_model.print_topics(-1):
            #print('Words: {}'.format(topic))

    def __filter_out_words(self):
        """
            filter documents to contain only __TOKENS_TO_KEEP tokens with 
            more than __MINIMUM_DOCUMENT_APPEARANCES,
            less than __MAXIMUM_DOCUMENT_APPEARANCE_FRACTION
        """
        self.__dictionary = gensim.corpora.Dictionary(self.__processed_docs)
        self.__dictionary.filter_extremes(no_below=self.__MINIMUM_DOCUMENT_APPEARANCES, no_above=self.__MAXIMUM_DOCUMENT_APPEARANCE_FRACTION, keep_n=self.__TOKENS_TO_KEEP)
        bow_corpus = [self.__dictionary.doc2bow(document) for document in self.__processed_docs]  # contains (word_id,times_appeard) tuples
        return bow_corpus

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
        return self.__stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
 