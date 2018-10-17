from FeatureExtractors.FeatureExtractor import FeatureExtractor
import numpy as np
import LDA
import gc

class TweetFeatureExtractor(FeatureExtractor):

    CHUNK_SIZE = 1000
    def get_tweet_features(self, hashtag):
        """
            stores the tweet related features
        """
        self.hashtag = hashtag
        self.tweets = self.dbHandler.getTweetsForHashtag(self.hashtag)

        tweet_features = {}
        #sentiment feature for tweet
        #tweet_features["tweet_sentiment"] = self.get_tweets_sentiment()
        #ratio features
        self.total_tweets = self.dbHandler.getNumOfTweets()
        # print("Extracting tweet ratio")
        # tweet_features["tweet_ratio"] = self.get_tweet_ratio()
        # print("Extracting author ratio")
        # tweet_features["author_ratio"] = self.get_author_ratio()
        # print("Extracting retweet ratio")
        # tweet_features["retweet_ratio"] = self.get_retweet_ratio()
        # print("Extracting mention ratio")
        # tweet_features["mention_ratio"] = self.get_mention_ratio()
        # print("Extracting url ratio")
        # tweet_features["url_ratio"] = self.get_url_ratio()
        #topic feature
        #tweet_features["topic"] = self.__get_topic()
        #word divergence distribution feature
        print("Extracting word divergence distribution")
        tweet_features["word_divergence_distribution"] = self.get_word_divergence()

        return tweet_features

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

        total_authors = set()
        chunk_size = self.CHUNK_SIZE
        skip = 0

        from tqdm import tqdm
        for _ in tqdm(range(0, self.total_tweets, chunk_size)):
            authors = self.dbHandler.getTweetAuthors(chunk_size, skip=skip)
            skip += chunk_size
            total_authors.update(authors)

        return len(hashtag_authors) / len(total_authors)

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

        total_retweets = 0
        chunk_size = self.CHUNK_SIZE
        skip = 0

        from tqdm import tqdm
        for _ in tqdm(range(0, self.total_tweets, chunk_size)):
            total_retweets += self.dbHandler.getRetweetsNum(chunk_size, skip=skip)
            skip += chunk_size

        return hashtag_retweets / total_retweets

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

        hashtag_clarity = []

        chunk_size = self.CHUNK_SIZE
        skip = 0

        from tqdm import tqdm
        for _ in tqdm(range(0, self.total_tweets, chunk_size)):
            text = self.dbHandler.getTweetTexts(chunk_size, skip=skip)
            skip += chunk_size

            divergence = self.get_divergence_for_chunk(text, hashtag_text)
            print(divergence)
            hashtag_clarity.append(divergence)

        return np.asarray(hashtag_clarity).mean()

    def get_divergence_for_chunk(self, text, hashtag_text):
        # calculate word frequencies
        tweet_dict = self.textToFreqDict(text)
        tweet_keys = list(tweet_dict.keys())
        tweet_list = (value / len(tweet_dict) for value in tweet_dict.values()) # () creates generator for ram efficiency

        word_dict = self.textToFreqDict(hashtag_text)
        word_list = (word_dict[key] / len(word_dict) if key in word_dict.keys() else 0 for key in tweet_keys)

        return self.KL(word_list, tweet_list)

    def KL(self, a, b):

        summary = 0
        for x, y in zip(a, b):
            if x != 0:
                summary += x * np.log(x/y)

        return summary#np.sum(np.where(a != 0, a * np.log(a / b), 0))

    def textToFreqDict(self, text):
        wordlist = np.asarray(text.split())
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

        total_urls = 0
        chunk_size = self.CHUNK_SIZE
        skip = 0

        from tqdm import tqdm
        for _ in tqdm(range(0, self.total_tweets, chunk_size)):
            tweets = self.dbHandler.getTweetsByNum(chunk_size, skip=skip)
            for tweet in tweets:
                if self.contains_urls(tweet):
                    total_urls += 1
            skip += chunk_size

        return hashtag_urls / total_urls

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

        total_mentions = 0
        chunk_size = self.CHUNK_SIZE
        skip = 0

        from tqdm import tqdm
        for _ in tqdm(range(0, self.total_tweets, chunk_size)):
            tweets = self.dbHandler.getTweetsByNum(chunk_size, skip=skip)
            for tweet in tweets:
                if self.contains_mentions(tweet):
                    total_mentions += 1
            print(total_mentions)
            skip += chunk_size

        return hashtag_mentions / total_mentions

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