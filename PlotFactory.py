import pandas as pd
from collections import Counter

class PlotFactory:
    def __init__(self, dbHandler, csv="features.csv"):
        self.dbHandler = dbHandler

        self.data = pd.read_csv(csv)
        self.hashtags = self.data["hashtag"].values

        self.get_tweets_for_hashtags()

    def get_tweets_for_hashtags(self):
        """
        :return: a dictionary with hashtags as key and a list of all the tweets contain the key hashtag as value.
        """

        self.hashtag_tweet_map = {}
        for hashtag in self.hashtags:
            self.hashtag_tweet_map[hashtag] = []

        for hashtag in self.hashtags:
            tweets = self.dbHandler.getTweetsForHashtag(hashtag)
            self.hashtag_tweet_map[hashtag].extend(tweets)
