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
         
        labels = ["text", "id"]
        self.__documents = pd.DataFrame.from_records(data, columns=labels)
        self.__stemmer = SnowballStemmer('english')
        self.__processed_docs = self.__documents["text"].map(self.__preprocess)
        self.__bow_corpus = self.__filter_out_words()

        self.__lda_bag_model = self.__lda_with_bag_of_words()
        self.__lda_tf_model = self.__lda_with_tf_idf()

    def predict_with_bag(self, tweetText):
        """
            predict the topic of a tweet using lda bag of words model
        """
        bow_vector = self.__dictionary.doc2bow(self.__preprocess(tweetText))
        prediction = sorted(self.__lda_bag_model[bow_vector], key=lambda tup: -1 * tup[1])

        index, score = prediction[0]

        return self.__lda_bag_model.print_topic(index, 5)

    def predict_with_tf_idf(self, tweetText):
        """
            predict the topic of a tweet using lda tf idf model
        """
        bow_vector = self.__dictionary.doc2bow(self.__preprocess(tweetText))
        prediction = sorted(self.__lda_tf_model[bow_vector], key=lambda tup: -1 * tup[1])

        index, score = prediction[0]

        return self.__lda_tf_model.print_topic(index, 5)

    def __lda_with_bag_of_words(self):
        """
            lda model using bag of words
        """

        lda_model = models.LdaMulticore(self.__bow_corpus, num_topics=20, id2word=self.__dictionary, passes=2,
                                        workers=2)

        return lda_model

    def __lda_with_tf_idf(self):
        """
            lda model using tf_idf
        """
        tfidf = models.TfidfModel(self.__bow_corpus)
        corpus_tfidf = tfidf[self.__bow_corpus]

        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20, id2word=self.__dictionary, passes=2,
                                                     workers=4)

        return lda_model_tfidf


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
 