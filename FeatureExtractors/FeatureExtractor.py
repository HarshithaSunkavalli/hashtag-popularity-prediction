from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import itertools
import operator
from collections import Counter

class FeatureExtractor:

    """attributes: 
        self.tweets
        self.hashtags
        self.dbHandler
        self.tweet_hashtag_map
    """
    K = 10
    TWEETS_TO_FETCH = 10000

    def __init__(self, dbHandler, k=False):
        self.dbHandler = dbHandler
        self.tweet_hashtag_map = {}

        if not k:
            self.tweets = self.dbHandler.getTweetsByNum(40)
        else:
            self.tweets = self.dbHandler.getTweetsFromTopK()#contains tweets for top 10 hashtags

        #run once to create collection
        #self.tweets, self.hashtags = self.__get_tweets_for_top_k_hashtags(self.K)
        #self.dbHandler.storeTopKTweets()

        self.hashtags = self.__get_hashtags()
        self.__sanitize_map()

    def __sanitize_map(self):
        """
        keep only top k hashtags
        self.hashtags already contain top k hashtags. But the map itself contains every hashtag that exists in each tweet.
        """
        for tweetId, hashtagList in self.tweet_hashtag_map.items():
            sanitized_hashtags = []
            for hashtag in hashtagList:
                if hashtag in self.hashtags:
                    sanitized_hashtags.append(hashtag)
            self.tweet_hashtag_map[tweetId] = sanitized_hashtags

    def __get_tweets_for_top_k_hashtags(self, k):
        """
        :param k: number of top hashtags
        :return: the tweets containing only top k hashtags
        """
        if k==0:
            self.tweets = []
            return


        tweets = self.dbHandler.getTweetsByNum(self.TWEETS_TO_FETCH)

        hashtags = []
        tweet_hashtag_map = {}
        for tweet in tweets:
            tweetHashtags = self.__get_hashtags_from_tweet(tweet, tweet_hashtag_map)
            tweetHashtags = [h["text"] for h in tweetHashtags]
            hashtags.extend(tweetHashtags)

        counter = Counter(hashtags)
        top_k = counter.most_common(k)#returns tuples of (hashtag, frequency)
        top_k = [h[0] for h in top_k]

        tweets_to_return = []
        for hashtag in top_k:
            tweets_to_return.extend(self.dbHandler.getTweetsForHashtag(hashtag))

        return  tweets_to_return, top_k

    def __get_hashtags(self):
        """
            Private method used to extract hashtags from the given tweets.
            self.tweet_hashtag_map: {'994633657141813248': [{'text': 'documentation', 'indices': [211, 225]}, {'text': 'parsingJSON', 'indices': [226, 238]}], '54691802283900928': [{'text': 'PGP', 'indices': [130, 134]}]}
            where key is the tweet id and value is a list of containing hashtags
            self.hashtags: [{'text': 'documentation', 'indices': [211, 225]}, {'text': 'parsingJSON', 'indices': [226, 238]}] and hashtags exist only once.
        """

        tweet_hashtag_map = {}
        hashtags = []
        for tweet in self.tweets:
            tweetHashtags = self.__get_hashtags_from_tweet(tweet, tweet_hashtag_map)
            for hashtag in tweetHashtags:
                if hashtag not in hashtags:#keep only a unique hashtag appearance
                    hashtags.append(hashtag)

        hashtag_text = [hashtag["text"] for hashtag in hashtags]
        counter = Counter(hashtag_text)
        top_k = counter.most_common(self.K)  # returns tuples of (hashtag, frequency)
        top_k = [h[0] for h in top_k]

        returned_hashtags = [ hashtag for hashtag in hashtags if hashtag["text"] in top_k]

        self.tweet_hashtag_map = tweet_hashtag_map

        return returned_hashtags

    def __get_hashtags_from_tweet(self, tweet, tweet_hashtag_map=None):
        """
            Private method used to extract hashtags from the given tweet.
            :return: the hashtags
        """
        hashtags = []
        # simple tweet
        if not tweet["truncated"]:
            first_level_potential_hashtags = tweet["entities"]["hashtags"]
            if first_level_potential_hashtags:
                hashtags.extend(first_level_potential_hashtags)
        # extended_tweet
        elif tweet["truncated"] and "retweeted_status" not in tweet:
            extended_potential_hashtags = tweet["extended_tweet"]["entities"]["hashtags"]
            if extended_potential_hashtags:
                hashtags.extend(extended_potential_hashtags)
        # retweet with no extended_tweet field
        # retweeted will be false if retweet has happened not using by using the button but the RT instead
        elif "retweeted_status" in tweet and (tweet["retweeted"]) and (not tweet["retweeted_status"]["truncated"]):
            retweeted_potential_hashtags = tweet["retweeted_status"]["entities"]["hashtags"]
            if retweeted_potential_hashtags:
                hashtags.extend(retweeted_potential_hashtags)
        # retweet with extended_tweet field
        elif ("retweeted_status" in tweet) and (tweet["retweeted"]) and (tweet["retweeted_status"]["truncated"]):
            retweeted_potential_hashtags = tweet["retweeted_status"]["extended_tweet"]["entities"]["hashtags"]
            if retweeted_potential_hashtags:
                hashtags.extend(retweeted_potential_hashtags)
        else:
            pass

        if tweet_hashtag_map is not None:
            tweet_hashtag_map[tweet["id_str"]] = hashtags

        return hashtags

    def get_tweets_sentiment(self):
        """
            returns a dictionary of (tweet_id, 2/1/0) attributes.
            0: negative
            1: neutral
            2: positive
        """
        analyzer = SentimentIntensityAnalyzer()

        tweet_sentiment = {}
        for tweet in self.tweets:
            tweet_sentiment[tweet["id_str"]] = ""

        for tweet in self.tweets:
            text = self.get_tweet_text(tweet)

            vs = analyzer.polarity_scores(text)
            sentiment = vs['compound']

            if sentiment >= 0.5:
                tweet_sentiment[tweet["id_str"]] = 2
            elif sentiment > -0.5 and sentiment < 0.5:
                tweet_sentiment[tweet["id_str"]] = 1
            else:
                tweet_sentiment[tweet["id_str"]] = 0

        return tweet_sentiment

    def get_tweet_text(self, tweet):
        """
            returns the text of the given tweet json
        """
        if "extended_tweet" in tweet:
            text = tweet["extended_tweet"]["full_text"]
        elif "retweeted_status" in tweet and tweet["retweeted"] and (not tweet["retweeted_status"]["truncated"]):
            text = tweet["retweeted_status"]["text"]
        elif "retweeted_status" in tweet and tweet["retweeted"] and (tweet["retweeted_status"]["truncated"]):
            text = tweet["retweeted_status"]["extended_tweet"]["full_text"]
        else:
            text = tweet["text"]

        return str(text)

    def get_sanitized_text(self, text):
        """
            Removes urls from given text
        """
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)

        return text

    def most_common(self, myList):
        """
            returns the most common element in a list
        """
        # get an iterable of (item, iterable) pairs
        sorted_list = sorted((x, i) for i, x in enumerate(myList))

        groups = itertools.groupby(sorted_list, key=operator.itemgetter(0))

        # auxiliary function to get "quality" for an item
        def _auxfun(g):
            item, iterable = g
            count = 0
            min_index = len(myList)
            for _, where in iterable:
                count += 1
                min_index = min(min_index, where)

            return count, -min_index

        # pick the highest-count/earliest item
        return max(groups, key=_auxfun)[0]