from .FeatureExtractor import FeatureExtractor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import itertools
import operator
import numpy as np
import LDA

class TweetFeatureExtractor(FeatureExtractor):

    def get_tweet_features(self):
        """
            returns the tweet related features
        """
        tweet_features = {}
        #sentiment feature
        tweet_features["tweet_sentiment"] = self.__get_tweets_sentiment()
        #ratio features
        tweet_features["tweet_ratio"] = self.__get_tweet_ratio()
        tweet_features["author_ratio"] = self.__get_author_ratio()
        tweet_features["retweet_ratio"] = self.__get_retweet_ratio()
        tweet_features["mention_ratio"] = self.__get_mention_ratio()
        tweet_features["url_ratio"] = self.__get_url_ratio()
        #topic feature
        tweet_features["tweet_topic"] = self.__get_tweet_topic()
        #word divergence distribution feature
        tweet_features["word_divergence_distribution"] = self.__get_word_divergence()

        return tweet_features

    def __get_tweets_sentiment(self):
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

    def __get_tweet_ratio(self):
        """
            returns a dictionary of (hashtag, tweet ratio) attributes presenting the ratio of tweets containing the specific hashtag
        """
        tweet_count = {}
        # initialize dictionary
        for hashtag in self.hashtags:
            tweet_count[hashtag["text"]] = 0

        # count appearances
        for _, hashtag_list in self.tweet_hashtag_map.items():
            for hashtag in hashtag_list:
                tweet_count[hashtag["text"]] += 1

        # extract actual ratio
        tweet_ratio = {hashtag: appearances / len(self.tweets) for hashtag, appearances in tweet_count.items()} if len(
            self.tweets) > 0 else {hashtag: 0.0 for hashtag in tweet_count.keys()}

        return tweet_ratio

    def __get_author_ratio(self):
        """
            returns a dictionary of (hashtag, author ratio) attributes presenting the ratio of authors who used the specific hashtag
        """
        authors = []
        author_track = {}
        # initialize dictionary
        for hashtag in self.hashtags:
            author_track[hashtag["text"]] = []

        # count authors
        for tweetId, hashtag_list in self.tweet_hashtag_map.items():
            authorId = self.__getAuthor(tweetId)
            for hashtag in hashtag_list:
                if authorId not in author_track[hashtag["text"]]:
                    author_track[hashtag["text"]].append(authorId)

                # count distinct authors
                if authorId not in authors:
                    authors.append(authorId)

        # map (hashtag, list of authors) to (hashtag, author count)
        author_count = {hashtag: len(author_list) for hashtag, author_list in author_track.items()}

        # extract actual ratio
        author_ratio = {hashtag: appearances / len(authors) for hashtag, appearances in author_count.items()} if len(
            authors) > 0 else {hashtag: 0.0 for hashtag in author_count.keys()}

        return author_ratio

    def __getAuthor(self, tweetId):
        """
            returns the author of the specific tweet
        """
        return self.dbHandler.getTweetById(tweetId)["user"]["id_str"]

    def __get_retweet_ratio(self):
        """
            returns a dictionary of (hashtag, retweet ratio) attributes presenting the ratio of retweets which contain the specific hashtag
        """

        total_retweets = 0
        retweet_count = {}
        for hashtag in self.hashtags:
            retweet_count[hashtag["text"]] = 0

        for tweetId, hashtag_list in self.tweet_hashtag_map.items():
            for hashtag in hashtag_list:
                if self.__is_retweet(tweetId):
                    retweet_count[hashtag["text"]] += 1
                    total_retweets += 1

        retweet_ratio = {hashtag: times_retweeted / total_retweets for hashtag, times_retweeted in
                         retweet_count.items()} if total_retweets > 0 else {hashtag: 0.0 for hashtag in
                                                                            retweet_count.keys()}

        return retweet_ratio

    def __is_retweet(self, tweetId):
        """
            returns true if tweet json contains retweeted status field which means that this is a retweet
        """
        tweet = self.dbHandler.getTweetById(tweetId)
        return "retweeted_status" in tweet

    def __get_word_divergence(self):
        """
            Returns a dictionary of (hashtag, clarity) attributes.
            Clarity is computed as the Kullback-Leibler word divergence distribution
        """

        hashtag_clarity = {}
        hashtag_text = {}
        for hashtag in self.hashtags:
            hashtag_clarity[hashtag["text"]] = 0
            hashtag_text[hashtag["text"]] = ""

        # create tweet text
        tweet_text = ""
        for tweet in self.tweets:
            text = self.get_tweet_text(tweet)
            text = self.get_sanitized_text(text)
            text += " "
            tweet_text += text
        # create hashtag text
        for tweetId, hashtag_list in self.tweet_hashtag_map.items():
            text = self.get_tweet_text(self.dbHandler.getTweetById(tweetId))
            text = self.get_sanitized_text(text)
            text += " "
            for hashtag in hashtag_list:
                hashtag_text[hashtag["text"]] += text

        # split text to words
        tweet_text_list = tweet_text.split()
        hashtag_text_list = {hashtag: text.split() for hashtag, text in hashtag_text.items()}

        # calculate word frequencies
        tweet_dict = self.__wordListToFreqDict(tweet_text_list)
        tweet_keys_sorted = sorted(tweet_dict)
        tweet_list = [tweet_dict[key] for key in tweet_keys_sorted]
        tweet_list = [value / len(tweet_list) for value in tweet_list]

        for hashtag, wordlist in hashtag_text_list.items():
            word_dict = self.__wordListToFreqDict(wordlist)
            # keep original length as long as it is going to change
            length = len(word_dict.keys())

            # populate word_dict so as to contain every word that tweet_dict contains
            for word in tweet_keys_sorted:
                if word not in word_dict.keys():
                    word_dict.update({word: 0})

            # words in word_list are in the same order as words in tweet_list
            word_list = [word_dict[key] for key in tweet_keys_sorted]
            word_list = [value / length for value in word_list]

            hashtag_clarity[hashtag] = self.__KL(word_list, tweet_list)

        return hashtag_clarity

    def __KL(self, a, b):
        a = np.asarray(a, dtype=np.float)
        b = np.asarray(b, dtype=np.float)

        return np.sum(np.where(a != 0, a * np.log(a / b), 0))

    def __wordListToFreqDict(self, wordlist):
        wordfreq = [wordlist.count(p) for p in wordlist]
        return dict(zip(wordlist, wordfreq))

    def __get_tweet_topic(self):
        """
            returns a dictionary of (tweetId: [(topic:probability),...]) attributes using Latent Dirichlet Allocation
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
        return tweet_topic

    def __get_hashtag_sentiment(self):
        """
            Returns a dictionary of (hashtag, positive/neutral/negative) attributes.
            The sentiment is extracted as the majority of distinct tweet sentiments
        """
        hashtag_sentiment = {}
        for hashtag in self.hashtags:
            hashtag_sentiment[hashtag["text"]] = []

        tweet_sentiment = self.__get_tweets_sentiment()
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

    def __get_url_ratio(self):
        """
            returns a dictionary of (hashtag, url ratio) attributes presenting the ratio of tweets which contain the specific hashtag as well as at least one url
        """
        total_urls = 0
        url_count = {}
        for hashtag in self.hashtags:
            url_count[hashtag["text"]] = 0

        for tweetId, hashtag_list in self.tweet_hashtag_map.items():
            for hashtag in hashtag_list:
                if self.__contains_urls(tweetId):
                    url_count[hashtag["text"]] += 1
                    total_urls += 1

        url_ratio = {hashtag: urls_contained_in / total_urls for hashtag, urls_contained_in in
                     url_count.items()} if total_urls > 0 else {hashtag: 0.0 for hashtag in url_count.keys()}

        return url_ratio

    def __contains_urls(self, tweetId):
        """
            returns true if tweet json contains at least one url
        """
        tweet = self.dbHandler.getTweetById(tweetId)
        return self.__contains_entities_element(tweet, "urls")

    def __get_mention_ratio(self):
        """
            returns a dictionary of (hashtag, mention ratio) attributes presenting the ratio of tweets which contain the specific hashtag as well as at least one mention
        """
        total_mentions = 0
        mention_count = {}
        for hashtag in self.hashtags:
            mention_count[hashtag["text"]] = 0

        for tweetId, hashtag_list in self.tweet_hashtag_map.items():
            for hashtag in hashtag_list:
                if self.__contains_mentions(tweetId):
                    mention_count[hashtag["text"]] += 1
                    total_mentions += 1

        mention_ratio = {hashtag: mentions_contained_in / total_mentions for hashtag, mentions_contained_in in
                         mention_count.items()} if total_mentions > 0 else {hashtag: 0.0 for hashtag in
                                                                            mention_count.keys()}

        return mention_ratio

    def __contains_mentions(self, tweetId):
        """
            returns true if tweet json contains at least one mention
        """
        tweet = self.dbHandler.getTweetById(tweetId)
        return self.__contains_entities_element(tweet, "user_mentions")

    def __contains_entities_element(self, tweet, element):
        """
            returns true if tweet json contains at least one of the given element
            element attribute indicate the existence of the specific element such as urls. But no element mean either no element attribute or element attribute of length 0
        """
        if not tweet["truncated"]:
            return (element in tweet["entities"]) and (tweet["entities"][element])  # not empty
        elif (tweet["truncated"]) and (
                "extended_tweet" in tweet):  # because there can be truncated retweets without containing an extended_tweet field
            return (element in tweet["extended_tweet"]["entities"]) and (tweet["extended_tweet"]["entities"][element])
        elif "retweeted_status" in tweet:
            if tweet["retweeted_status"]["truncated"]:
                # we care both for the retweet and the original tweet (which is truncated)
                return (
                        ((element in tweet["entities"]) and (tweet["entities"][element])) or
                        ((element in tweet["retweeted_status"]["extended_tweet"]["entities"]) and (
                        tweet["retweeted_status"]["extended_tweet"]["entities"][element]))
                )
            else:
                # original tweet is not truncated
                return (
                        ((element in tweet["entities"]) and (tweet["entities"][element])) or
                        ((element in tweet["retweeted_status"]["entities"]) and (
                        tweet["retweeted_status"]["entities"][element]))
                )
        else:
            return False