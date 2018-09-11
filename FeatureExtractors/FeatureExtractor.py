from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import itertools
import operator

class FeatureExtractor:

    """attributes: 
        self.tweets
        self.hashtags
        self.dbHandler
        self.tweet_hashtag_map
    """

    def __init__(self, dbHandler):
        self.dbHandler = dbHandler
        self.tweets = list(self.dbHandler.getTweets())
        self.tweet_hashtag_map = {}
        self.hashtags = self.__get_hashtags()

    def __get_hashtags(self):
        """
            Private method used to extract hashtags from the given tweets.
            self.tweet_hashtag_map: {'994633657141813248': [{'text': 'documentation', 'indices': [211, 225]}, {'text': 'parsingJSON', 'indices': [226, 238]}], '54691802283900928': [{'text': 'PGP', 'indices': [130, 134]}]}
            where key is the tweet id and value is a list of containing hashtags
            self.hashtags: [{'text': 'documentation', 'indices': [211, 225]}, {'text': 'parsingJSON', 'indices': [226, 238]}] and hashtags exist only once.
        """

        tweet_hashtag_map = {}
        for tweet in self.tweets:
            self.__get_hashtags_from_tweet(tweet, tweet_hashtag_map)

        hashtags = []
        for hashtag_list in tweet_hashtag_map.values():
            for hashtag in hashtag_list:
                if hashtag not in hashtags:
                    hashtags.append(hashtag)

        self.tweet_hashtag_map = tweet_hashtag_map

        return hashtags

    def __get_hashtags_from_tweet(self, tweet, tweet_hashtag_map):
        """
            Private method used to extract hashtags from the given tweet.
        """
        hashtags = []
        # simple tweet
        if not tweet["truncated"]:
            first_level_potential_hashtags = tweet["entities"]["hashtags"]
            if first_level_potential_hashtags:
                hashtags.extend(first_level_potential_hashtags)

        # extended_tweet
        if tweet["truncated"] and "retweeted_status" not in tweet:
            extended_potential_hashtags = tweet["extended_tweet"]["entities"]["hashtags"]
            if extended_potential_hashtags:
                hashtags.extend(extended_potential_hashtags)

        # retweet with no extended_tweet field
        if "retweeted_status" in tweet and (not tweet["retweeted_status"]["truncated"]):
            retweeted_potential_hashtags = tweet["retweeted_status"]["entities"]["hashtags"]
            if retweeted_potential_hashtags:
                hashtags.extend(retweeted_potential_hashtags)

        # retweet with extended_tweet field
        if ("retweeted_status" in tweet) and (tweet["retweeted_status"]["truncated"]):
            retweeted_potential_hashtags = tweet["retweeted_status"]["extended_tweet"]["entities"]["hashtags"]
            if retweeted_potential_hashtags:
                hashtags.extend(retweeted_potential_hashtags)

        tweet_hashtag_map[tweet["id_str"]] = hashtags

    def get_tweets_sentiment(self):
        """
            returns a dictionary of (tweet_id, positive/neutral/negative) attributes.
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
                tweet_sentiment[tweet["id_str"]] = "positive"
            elif sentiment > -0.5 and sentiment < 0.5:
                tweet_sentiment[tweet["id_str"]] = "neutral"
            else:
                tweet_sentiment[tweet["id_str"]] = "negative"

        return tweet_sentiment

    def get_tweet_text(self, tweet):
        """
            returns the text of the given tweet json
        """
        if "extended_tweet" in tweet:
            text = tweet["extended_tweet"]["full_text"]
        elif "retweeted_status" in tweet and (not tweet["retweeted_status"]["truncated"]):
            text = tweet["retweeted_status"]["text"]
        elif "retweeted_status" in tweet and (tweet["retweeted_status"]["truncated"]):
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