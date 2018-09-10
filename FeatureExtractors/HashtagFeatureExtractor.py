from .FeatureExtractor import FeatureExtractor
import numpy as np
import itertools
import operator
import re

class HashtagFeatureExtractor(FeatureExtractor):

    __COOCCURANCE_THRESHOLD = 0.4
    __LOCATION_THRESHOLD = 0.25

    def get_hashtag_features(self):
        """
            returns the hashtag related features
        """

        hashtag_features = {}
        # char length feature
        hashtag_features["char_length"] = self.__get_hashtags_length()
        # orthography related features
        hashtag_features["contains_digits"] = self.__get_hashtags_contain_digits()
        hashtag_features["all_caps"] = self.__get_hashtags_all_caps()
        hashtag_features["any_caps"] = self.__get_hashtags_any_caps()
        hashtag_features["no_caps"] = self.__get_hashtags_no_caps()
        hashtag_features["special_signals"] = self.__get_hashtags_special_signals()
        # co-occurance feature
        hashtag_features["cooccurance"] = self.__get_hashtags_cooccurance()
        # location feature
        hashtag_features["location"] = self.__get_hashtags_location()
        # hashtag sentiment feature
        hashtag_features["hashtag_sentiment"] = self.__get_hashtag_sentiment()

        return hashtag_features

    def __get_hashtags_location(self):
        """
            finds the location of each hashtag inside the corresponding tweet text
            returns a dictionary of (hashtag, ratio) attributes.
            The ratio is calculated as the mean of the distinct hashtag ratios of each tweet they appear in
        """

        hashtag_ratio = {}
        for hashtag in self.hashtags:
            hashtag_ratio[hashtag["text"]] = []

        for tweet_hashtag in self.tweet_hashtag_map.items():
            tweet_id = tweet_hashtag[0]
            tweet = self.dbHandler.getTweetById(tweet_id)

            text = self.get_tweet_text(tweet)
            word_list = text.split()

            hashtag_list = tweet_hashtag[1]
            for hashtag in hashtag_list:
                pattern = "#{}".format(hashtag["text"])

                position = word_list.index(pattern)
                ratio = position / len(word_list)
                hashtag_ratio[hashtag["text"]].append(ratio)

        hashtag_ratio = {hashtag: np.array(ratio_list).mean() for hashtag, ratio_list in hashtag_ratio.items()}

        return hashtag_ratio

    def __get_hashtags_cooccurance(self):
        """
            returns a dictionary of (hashtag, true/false) attributes.
            True is given if more than 40% of the specific hashtag occurences are collocated with other hashtags
        """

        appearance_counter = {}
        cooccurance_counter = {}

        # initialize counters
        for hashtag in self.hashtags:
            appearance_counter[hashtag["text"]] = 0
            cooccurance_counter[hashtag["text"]] = 0

        for hashtag_list in self.tweet_hashtag_map.values():
            if len(hashtag_list) == 1:  # no coexisting hashtags in this list
                appearance_counter[hashtag_list[0]["text"]] += 1
            else:  # only coexisting hashtags in this list
                for hashtag in hashtag_list:
                    appearance_counter[hashtag["text"]] += 1
                    cooccurance_counter[hashtag["text"]] += 1

        ###calculate ratio so as to consider 40% threshold
        ratio_hashtag_cooccurance = {}
        for hashtag, value in appearance_counter.items():
            ratio_hashtag_cooccurance[hashtag] = cooccurance_counter[hashtag] / float(value)

        for hashtag, value in ratio_hashtag_cooccurance.items():
            ratio_hashtag_cooccurance[hashtag] = True if ratio_hashtag_cooccurance[
                                                             hashtag] >= self.__COOCCURANCE_THRESHOLD else False

        return ratio_hashtag_cooccurance

    def __get_hashtags_special_signals(self):
        """
            returns a dictionary of (hashtag, true/false) attributes whether they contain special signals, such as gooood or !!!!, or not.
            3 or more consecutive letters or symbols required
        """

        special_signals = {}
        for hashtag in self.hashtags:
            temp_list = re.findall(r'((\w)\2{2,})', hashtag["text"])
            special_signals[hashtag["text"]] = True if len(temp_list) else False

        return special_signals

    def __get_hashtags_no_caps(self):
        """
            returns a dictionary of (hashtag, true/false) attributes whether they contain only lowercase letters or not
        """
        no_caps = {}
        for hashtag in self.hashtags:
            no_caps[hashtag["text"]] = hashtag["text"].islower()

        return no_caps

    def __get_hashtags_any_caps(self):
        """
            returns a dictionary of (hashtag, true/false) attributes whether they contain any capital letters or not
        """
        any_caps = {}
        for hashtag in self.hashtags:
            any_caps[hashtag["text"]] = any(char.isupper() for char in hashtag["text"])

        return any_caps

    def __get_hashtags_all_caps(self):
        """
            returns a dictionary of (hashtag, true/false) attributes whether they contain only capital letters or not
        """
        all_caps = {}
        for hashtag in self.hashtags:
            all_caps[hashtag["text"]] = hashtag["text"].isupper()

        return all_caps

    def __get_hashtags_contain_digits(self):
        """
            returns a dictionary of (hashtag, true/false) attributes whether they contain digits or not
        """
        contain_digits = {}
        for hashtag in self.hashtags:
            contain_digits[hashtag["text"]] = any(char.isdigit() for char in hashtag["text"])

        return contain_digits

    def __get_hashtags_length(self):
        """
            returns a dictionary of (hashtag, hashtagLength) attributes
        """
        char_length = {}
        for hashtag in self.hashtags:
            char_length[hashtag["text"]] = hashtag["indices"][1] - hashtag["indices"][0] - 1

        return char_length

    def __get_hashtag_sentiment(self):
        """
            Returns a dictionary of (hashtag, positive/neutral/negative) attributes.
            The sentiment is extracted as the majority of distinct tweet sentiments
        """
        hashtag_sentiment = {}
        for hashtag in self.hashtags:
            hashtag_sentiment[hashtag["text"]] = []

        tweet_sentiment = self.get_tweets_sentiment()
        for tweetId, sentiment in tweet_sentiment.items():
            hashtag_list = self.tweet_hashtag_map[tweetId]  # get tweet specific hashtags
            for hashtag in hashtag_list:
                hashtag_sentiment[hashtag["text"]].append(sentiment)

        def __most_common(sentiment_list):
            """
                returns the most common element in a list
            """
            # get an iterable of (item, iterable) pairs
            sorted_list = sorted((x, i) for i, x in enumerate(sentiment_list))

            groups = itertools.groupby(sorted_list, key=operator.itemgetter(0))

            # auxiliary function to get "quality" for an item
            def _auxfun(g):
                item, iterable = g
                count = 0
                min_index = len(sentiment_list)
                for _, where in iterable:
                    count += 1
                    min_index = min(min_index, where)

                return count, -min_index

            # pick the highest-count/earliest item
            return max(groups, key=_auxfun)[0]

        hashtag_sentiment = {hashtag: __most_common(sentiment_list) for hashtag, sentiment_list in
                             hashtag_sentiment.items()}

        return hashtag_sentiment