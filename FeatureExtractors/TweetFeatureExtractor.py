from FeatureExtractors.FeatureExtractor import FeatureExtractor
import numpy as np
import LDA
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TweetFeatureExtractor(FeatureExtractor):

    def precalculateValues(self):
        self.total_tweets = self.dbHandler.getNumOfTweets()
        print("Extracting total attributes")
        self.total_retweets, self.total_authors, self.total_urls, self.total_mentions, self.total_tweet_list, self.total_tweet_keys = self.get_total_attributes()


    def get_tweet_features(self, hashtag):
        """
            stores the tweet related features
        """
        self.hashtag = hashtag
        self.tweets = self.dbHandler.getTweetsForHashtag(self.hashtag)

        tweet_features = {}
        #sentiment feature for tweet
        #tweet_features["tweet_sentiment"] = self.get_tweets_sentiment()
        # topic feature
        # tweet_features["topic"] = self.__get_topic()
        #ratio features
        tweet_features["tweet_ratio"] = self.get_tweet_ratio()
        tweet_features["author_ratio"] = self.get_author_ratio()
        tweet_features["retweet_ratio"] = self.get_retweet_ratio()
        tweet_features["mention_ratio"] = self.get_mention_ratio()
        tweet_features["url_ratio"] = self.get_url_ratio()
        # word divergence distribution feature
        tweet_features["word_divergence_distribution"] = self.get_word_divergence()

        return tweet_features

    def get_total_attributes(self):
        """
        Calculate 4 attributes for all tweets in database.
        Attributes: total number of retweets, urls, mentions and authors
        They will be used as denominators in specific ratio extractions
        """
        total_retweets = 0
        total_urls = 0
        total_mentions = 0

        chunk_size = self.CHUNK_SIZE
        skip = 0

        authors = set()
        total_tweet_list = []
        total_tweet_keys = []
        from tqdm import tqdm
        for _ in tqdm(range(0, self.total_tweets, chunk_size)):
            tweets = self.dbHandler.getTweetsByNum(chunk_size, skip=skip)
            for tweet in tweets:
                if self.is_retweet(tweet):
                    total_retweets += 1
                if self.contains_mentions(tweet):
                    total_urls += 1
                if self.contains_mentions(tweet):
                    total_mentions += 1

                authors.add(self.get_author(tweet))

            text = self.dbHandler.getTweetTexts(chunk_size, skip=skip)

            tweet_dict = self.textToFreqDict(text)
            tweet_keys = list(tweet_dict.keys())
            tweet_list = [value / len(tweet_dict) for value in tweet_dict.values()]  # () creates generator for ram efficiency
            total_tweet_keys.append(tweet_keys)
            total_tweet_list.append(tweet_list)

            skip += chunk_size

        total_authors = len(authors)



        return total_retweets, total_authors, total_urls, total_mentions, total_tweet_list, total_tweet_keys

    def get_tweet_ratio(self):
        """
            returns the ratio of tweets containing the specific hashtag
        """
        total_tweets = self.dbHandler.getNumOfTweets()

        return len(self.tweets) / total_tweets

    def get_author_ratio(self):
        """
            returns the ratio of authors who used the specific hashtag
        """
        hashtag_authors = set()
        for tweet in self.tweets:
            author = self.get_author(tweet)
            hashtag_authors.add(author)

        return len(hashtag_authors) / self.total_authors

    def get_author(self, tweet):
        """
            returns the author of the specific tweet
        """
        return tweet["user"]["id_str"]

    def get_retweet_ratio(self):
        """
            returns the ratio of retweets which contain the specific hashtag
        """
        hashtag_retweets = 0
        for tweet in self.tweets:
            if self.is_retweet(tweet):
                hashtag_retweets += 1

        return hashtag_retweets / self.total_retweets

    def is_retweet(self, tweet):
        """
            returns true if tweet json contains retweeted status field which means that this is a retweet
        """
        return "retweeted_status" in tweet

    def get_word_divergence(self):
        """
            Returns a dictionary of (hashtag, clarity) attributes.
            Clarity is computed as the Kullback-Leibler word divergence distribution
        """
        # create hashtag text
        hashtag_text = ""
        for tweet in self.tweets:
            text = self.get_tweet_text(tweet)
            text = self.get_sanitized_text(text)
            text += " "
            hashtag_text += text

        hashtag_word_dict = self.textToFreqDict(hashtag_text)

        hashtag_clarity = []
        for tweet_list, tweet_keys in zip(self.total_tweet_list, self.total_tweet_keys):
            divergence = self.get_divergence_for_chunk(tweet_list, tweet_keys, hashtag_word_dict)
            hashtag_clarity.append(divergence)

        return np.asarray(hashtag_clarity).mean()

    def get_divergence_for_chunk(self, tweetList, tweetKeys, hashtag_word_dict):
        word_list = (hashtag_word_dict[key] / len(hashtag_word_dict) if key in hashtag_word_dict.keys() else 0 for key in tweetKeys)
        return self.KL(word_list, tweetList)

    def KL(self, a, b):

        summary = 0
        for x, y in zip(a, b):
            if x != 0:
                summary += x * np.log(x/y)

        return summary#np.sum(np.where(a != 0, a * np.log(a / b), 0))

    def textToFreqDict(self, text):
        """
        Tokenizes text.
        Removes stopwords.
        Stems and lemmatizes
        Convert to lowercase
        :param text:
        :return: word frequency dictionary
        """
        stemmer = PorterStemmer()
        wordnet_lemmatizer = WordNetLemmatizer()
        wordlist = word_tokenize(text)

        stop_words = set(stopwords.words('english'))

        wordlist = [word for word in wordlist if not word in stop_words]
        wordlist = [stemmer.stem(word) for word in wordlist]
        wordlist = [wordnet_lemmatizer.lemmatize(word) for word in wordlist]
        wordlist = [word.lower() for word in wordlist]

        unique, counts = np.unique(wordlist, return_counts=True)
        return dict(zip(unique, counts))

    def __get_topic(self):
        """
            returns a dictionary of (hashtag: topic) attributes using Latent Dirichlet Allocation
        """
        tweet_topic = {}
        tweet_data = []
        for tweet in self.tweets:
            tweet_topic[tweet["id_str"]] = ""

            text = self.get_tweet_text(tweet)
            tweet_data.append((text, tweet["id_str"]))

        lda = LDA.LDA(tweet_data)

        for tweet in self.tweets:
            text = self.get_tweet_text(tweet)
            tweet_topic[tweet["id_str"]] = lda.predict_with_bag(text)
            # tweet_topic[tweet["id_str"]] = lda.predict_with_tf_idf(text)

        hashtag_topic = {}
        for hashtag in self.hashtags:
            hashtag_topic[hashtag["text"]] = []

        for tweet, hashtagList in self.tweet_hashtag_map.items():
            for hashtag in hashtagList:
                hashtag_topic[hashtag["text"]].append(tweet_topic[tweet])

        #hashtag topic is the the topic of the majority of hashtag's tweets
        hashtag_topic = {hashtag: self.most_common(l) for hashtag, l in
                             hashtag_topic.items()}
        return hashtag_topic

    def get_url_ratio(self):
        """
            returns the ratio of tweets which contain the specific hashtag as well as at least one url
        """
        hashtag_urls = 0
        for tweet in self.tweets:
            if self.contains_urls(tweet):
                hashtag_urls += 1

        return hashtag_urls / self.total_urls

    def contains_urls(self, tweet):
        """
            returns true if tweet json contains at least one url
        """
        return self.contains_entities_element(tweet, "urls")

    def get_mention_ratio(self):
        """
            returns the ratio of tweets which contain the specific hashtag as well as at least one mention
        """
        hashtag_mentions = 0
        for tweet in self.tweets:
            if self.contains_mentions(tweet):
                hashtag_mentions += 1

        return hashtag_mentions / self.total_mentions

    def contains_mentions(self, tweet):
        """
            returns true if tweet json contains at least one mention
        """
        return self.contains_entities_element(tweet, "user_mentions")

    def contains_entities_element(self, tweet, element):
        """
            returns true if tweet json contains at least one of the given element
            element attribute indicate the existence of the specific element such as urls. But no element mean either no element attribute or element attribute of length 0
        """
        if not tweet["truncated"]:
            return (element in tweet["entities"]) and (tweet["entities"][element])  # not empty
        elif (tweet["truncated"]) and (
                "extended_tweet" in tweet):  # because there can be truncated retweets without containing an extended_tweet field
            return tweet["extended_tweet"]["entities"][element]
        elif "retweeted_status" in tweet and tweet["retweeted"]:
            if tweet["retweeted_status"]["truncated"]:
                # we care both for the retweet and the original tweet (which is truncated)
                return (
                        (tweet["entities"][element]) or
                        (tweet["retweeted_status"]["extended_tweet"]["entities"][element])
                )
            else:
                # original tweet is not truncated
                return (
                        (tweet["entities"][element]) or
                        (tweet["retweeted_status"]["entities"][element])
                )
        else:
            return False