from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import itertools
import operator
import pandas as pd

class FeatureExtractor:

    """attributes: 
        self.tweets
        self.hashtags
        self.dbHandler
        self.tweet_hashtag_map
    """
    K = 10
    TWEETS_TO_FETCH = 10000
    CHUNK_SIZE = 1000

    def __init__(self, dbHandler=None, featureExtractor=None):

        if featureExtractor:
            self.dbHandler = featureExtractor.dbHandler
            self.tweets = featureExtractor.tweets
        else:
            self.dbHandler = dbHandler
            self.tweets = []

    def get_hashtags_from_tweet(self, tweet):
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

        hashtags = [hashtag["text"] for hashtag in hashtags]

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

    def create_hashtag_csv(self, ioHandler):
        """
        :param k: number of top hashtags
        """

        chunk_size = self.CHUNK_SIZE
        skip = 0

        total_tweets = self.dbHandler.getNumOfTweets()
        hashtags = set()
        from tqdm import tqdm
        for _ in tqdm(range(0, total_tweets, chunk_size)):
            tweets = (el for el in self.dbHandler.getTweetsByNum(chunk_size, skip=skip))  # create generator from list
            temp = 0
            for tweet in tweets:
                temp += len(self.get_hashtags_from_tweet(tweet))
                hashtags.update(self.get_hashtags_from_tweet(tweet))

            skip += chunk_size

        ioHandler.writeListToCSV(hashtags)

    def create_top_k_csv(self, ioHandler, input="hashtags.csv", output="top_k.csv"):

        hashtags = pd.read_csv(input)["hashtag"]
        from tqdm import tqdm
        appearances = [len(self.dbHandler.getTweetsForHashtag(hashtag)) for hashtag in tqdm(hashtags)]

        hashtags, appearances = (list(t) for t in zip(*sorted(zip(hashtags, appearances), key=operator.itemgetter(1), reverse=True)))

        hashtags = hashtags[:10]

        ioHandler.writeListToCSV(hashtags, my_csv=output)


