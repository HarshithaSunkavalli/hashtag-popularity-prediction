from .FeatureExtractor import FeatureExtractor
import numpy as np
import re
import time
from datetime import datetime, timedelta

class HashtagFeatureExtractor(FeatureExtractor):

    __COOCCURANCE_THRESHOLD = 0.4
    __LOCATION_THRESHOLD = 0.25

    def get_hashtag_features(self, hashtag):
        """
            stores the hashtag related features
        """
        self.hashtag = hashtag
        self.tweets = self.dbHandler.getTweetsForHashtag(self.hashtag)

        hashtag_features = {}
        # char length feature
        print("Extracting char length")
        hashtag_features["char_length"] = self.get_hashtags_length()
        # orthography related features
        print("Extracting orthography")
        hashtag_features["contains_digits"] = self.get_hashtags_contain_digits()
        hashtag_features["all_caps"] = self.get_hashtags_all_caps()
        hashtag_features["any_caps"] = self.get_hashtags_any_caps()
        hashtag_features["no_caps"] = self.get_hashtags_no_caps()
        hashtag_features["special_signals"] = self.get_hashtags_special_signals()
        # co-occurance feature
        print("Extracting co-occurance")
        hashtag_features["cooccurance"] = self.get_hashtags_cooccurance()
        # location feature
        print("Extracting location")
        hashtag_features["location"] = self.get_hashtags_location()
        # hashtag sentiment feature
        print("Extracting sentiment")
        hashtag_features["hashtag_sentiment"] = self.get_hashtag_sentiment()
        # hashtag popularity feature
        print("Extracting popularity")
        hashtag_features["popularity"] = self.get_hashtag_popularity()
        # hashtag time series features
        print("Extracting creation time and lifespan")
        hashtag_features["created_at"], hashtag_features["lifespan"] = self.get_hashtags_created_at_and_lifespan()

        return hashtag_features


    def get_hashtags_created_at_and_lifespan(self):
        """
        :param hashtag: the hashtag to calculate creation time and lifespan
        :return: the creation time (as a distance from Unix time: January 1, 1970) and lifespan (in seconds) of given hashtag
        """
        hashtag_creation_time = []
        for tweet in self.tweets:
            created_at = tweet["created_at"]
            hashtag_creation_time.append(created_at)

        if len(hashtag_creation_time) == 1:
            return time.mktime(hashtag_creation_time[0].timetuple()), timedelta().total_seconds()#return the same object as the else statement
        else:
            oldest = min(hashtag_creation_time)
            newest = max(hashtag_creation_time)
            lifespan = newest - oldest
            return time.mktime(oldest.timetuple()), lifespan.total_seconds()

    def get_hashtag_popularity(self):
        """
        :return: the number of tweets that the hashtag exists in
        """

        return len(self.tweets)

    def get_hashtag_sentiment(self):
        """
            The sentiment is extracted as the majority of distinct tweet sentiments
            0: negative
            1: neutral
            2: positive
        """
        hashtag_sentiment = []
        tweet_sentiment = self.get_tweets_sentiment()
        for _, sentiment in tweet_sentiment.items():
            hashtag_sentiment.append(sentiment)

        return self.most_common(hashtag_sentiment)


    def get_hashtags_location(self):
        """
            finds the location of hashtag inside the corresponding tweet text
            The ratio is calculated as the mean of the distinct hashtag ratios of each tweet it appears in
        """

        hashtag_ratio = []

        for tweet in self.tweets:

            text = self.get_tweet_text(tweet)
            text = re.sub(r"[^\w\s#]", ' ', text)# replace special characters with a space
            word_list = text.split()

            pattern = "#{}".format(self.hashtag)

            position = word_list.index(pattern)
            ratio = position / len(word_list)
            hashtag_ratio.append(ratio)

        return np.array(hashtag_ratio).mean()

    def get_hashtags_cooccurance(self):
        """
            returns True if more than 40% of the specific hashtag occurences are collocated with other hashtags
            1: true
            0: false
        """

        appearance_counter = len(self.tweets)
        cooccurance_counter = 0

        # initialize counters
        for tweet in self.tweets:
            hashtags = self.get_hashtags_from_tweet(tweet)
            if len(hashtags) > 1:
                cooccurance_counter += 1

        ratio = cooccurance_counter / appearance_counter

        return 1 if ratio >= self.__COOCCURANCE_THRESHOLD else 0

    def get_hashtags_special_signals(self):
        """
            returns whether the hashtag contains special signals, such as gooood or !!!!, or not.
            3 or more consecutive letters or symbols required
            1: true
            0: false
        """

        temp_list = re.findall(r'((\w)\2{2,})', self.hashtag)

        return 1 if len(temp_list) else 0

    def get_hashtags_no_caps(self):
        """
            returns whether the hashtag contains only lowercase letters or not
            1: true
            0: false
        """

        return 1 if self.hashtag.islower() else 0

    def get_hashtags_any_caps(self):
        """
            returns whether the hashtag contains any capital letters or not
            1: true
            0: false
        """

        return 1 if any(char.isupper() for char in self.hashtag) else 0

    def get_hashtags_all_caps(self):
        """
            returns whether the hashtag contains only capital letters or not
            1: true
            0: false
        """

        return 1 if self.hashtag.isupper() else 0

    def get_hashtags_contain_digits(self):
        """
            returns whether the hashtag contains digits or not.
            1: true
            0: false
        """

        return 1 if any(char.isdigit() for char in self.hashtag) else 0

    def get_hashtags_length(self):
        """
            returns the hashtag length
        """

        return len(self.hashtag)