class FeatureExtractor:

    def __init__(self, tweets):
        self.tweets = tweets

    def get_hashtag_features(self):
        """
            returns  the following hashtag related features:
                character length,
        """
        self.__get_hashtags()

        hashtag_features = {}
        hashtag_features["char_length"] = self.__get_hashtags_length()
        hashtag_features["contains_digits"] = self.__get_hashtags_contain_digits()

        return hashtag_features
        
    
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

    def __get_hashtags(self):
        """
            Private method used to extract hashtags from the given tweets.
            hashtags list form: [{'text': 'documentation', 'indices': [211, 225]}, {'text': 'parsingJSON', 'indices': [226, 238]}]
        """

        hashtags = []
        for tweet in self.tweets:
            #simple tweet
            first_level_potential_hashtags = tweet["entities"]["hashtags"]
            if len(first_level_potential_hashtags) > 0:
                hashtags.extend(first_level_potential_hashtags)

            #extended_tweet
            if "extended_tweet" in tweet:
                extended_potential_hashtags = tweet["extended_tweet"]["entities"]["hashtags"]
                if len(extended_potential_hashtags) > 0:
                    hashtags.extend(extended_potential_hashtags)

            #retweet
            if "retweeted_status" in tweet:
                retweeted_potential_hashtags = tweet["retweeted_status"]["entities"]["hashtags"]
                if len(retweeted_potential_hashtags) > 0:
                    hashtags.extend(retweeted_potential_hashtags)
        
        self.hashtags = hashtags

        
        
                