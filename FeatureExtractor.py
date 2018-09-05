import re
import DbHandler
import LDA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import itertools
import operator

class FeatureExtractor:

    __COOCCURANCE_THRESHOLD = 0.4
    __LOCATION_THRESHOLD = 0.25
    """attributes: 
        self.tweets
        self.tweet_hashtag_map
        self.hashtags
        self.dbHandler
    """

    def __init__(self, dbHandler):
        self.dbHandler = dbHandler
        self.tweets = list(self.dbHandler.getTweets())

    def get_hashtag_features(self):
        """
            returns the hashtag related features
        """
        self.__get_hashtags()
        
        hashtag_features = {}
        #char length feature
        hashtag_features["char_length"] = self.__get_hashtags_length()
        #orthography related features
        hashtag_features["contains_digits"] = self.__get_hashtags_contain_digits()
        hashtag_features["all_caps"] = self.__get_hashtags_all_caps()
        hashtag_features["any_caps"] = self.__get_hashtags_any_caps()
        hashtag_features["no_caps"] = self.__get_hashtags_no_caps()
        hashtag_features["special_signals"] = self.__get_hashtags_special_signals()
        #co-occurance feature
        hashtag_features["cooccurance"] = self.__get_hashtags_cooccurance()
        #location feature
        hashtag_features["location"] = self.__get_hashtags_location()
        #hashtag sentiment feature
        hashtag_features["hashtag_sentiment"] = self.__get_hashtag_sentiment()
        
        return hashtag_features

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

        return tweet_features
    
    def __get_tweet_topic(self):
        """
            returns a dictionary of (tweetId: [(topic:probability),...]) attributes using Latent Dirichlet Allocation
        """
        tweet_data = []
        index = 0
        for tweet in self.tweets:
            text = self.__get_tweet_text(tweet)
            tweet_data.append((text, index))
            index += 1
        
        lda = LDA.LDA(tweet_data)
        print(lda.get_docs())
    
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
            hashtag_list = self.tweet_hashtag_map[tweetId] #get tweet specific hashtags
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

        hashtag_sentiment = {hashtag: __most_common(sentiment_list) for hashtag, sentiment_list in hashtag_sentiment.items()}
        
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

        url_ratio = {hashtag: urls_contained_in / total_urls for hashtag, urls_contained_in in url_count.items()} if total_urls > 0 else {hashtag: 0.0 for hashtag in url_count.keys()}

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
      
        mention_ratio = {hashtag: mentions_contained_in / total_mentions for hashtag, mentions_contained_in in mention_count.items()} if total_mentions > 0 else {hashtag: 0.0 for hashtag in mention_count.keys()}

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
            return (element in tweet["entities"]) and (tweet["entities"][element])#not empty
        elif (tweet["truncated"]) and ("extended_tweet" in tweet): #because there can be truncated retweets without containing an extended_tweet field
            return (element in tweet["extended_tweet"]["entities"])  and (tweet["extended_tweet"]["entities"][element])
        elif "retweeted_status" in tweet: 
            if tweet["retweeted_status"]["truncated"]:
                #we care both for the retweet and the original tweet (which is truncated)
                return (
                        ((element in tweet["entities"]) and (tweet["entities"][element])) or 
                        ((element in tweet["retweeted_status"]["extended_tweet"]["entities"]) and (tweet["retweeted_status"]["extended_tweet"]["entities"][element]))
                    )
            else:
                #original tweet is not truncated
                return (
                        ((element in tweet["entities"]) and (tweet["entities"][element])) or 
                        ((element in tweet["retweeted_status"]["entities"]) and (tweet["retweeted_status"]["entities"][element]))
                    )
        else:
            return False

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
        
        retweet_ratio = {hashtag: times_retweeted / total_retweets for hashtag, times_retweeted in retweet_count.items()} if total_retweets > 0 else {hashtag: 0.0 for hashtag in retweet_count.keys()}

        return retweet_ratio

    def __is_retweet(self, tweetId):
        """
            returns true if tweet json contains retweeted status field which means that this is a retweet
        """
        tweet = self.dbHandler.getTweetById(tweetId)
        return "retweeted_status" in tweet 

    def __get_author_ratio(self):
        """
            returns a dictionary of (hashtag, author ratio) attributes presenting the ratio of authors who used the specific hashtag
        """   
        authors = []
        author_track = {}
        #initialize dictionary
        for hashtag in self.hashtags:
            author_track[hashtag["text"]] = []
        
        #count authors
        for tweetId, hashtag_list in self.tweet_hashtag_map.items():
            authorId = self.__getAuthor(tweetId)
            for hashtag in hashtag_list:
                if authorId not in author_track[hashtag["text"]]:
                    author_track[hashtag["text"]].append(authorId)
                
                #count distinct authors
                if authorId not in authors:
                    authors.append(authorId)

        #map (hashtag, list of authors) to (hashtag, author count)
        author_count = {hashtag: len(author_list) for hashtag, author_list in author_track.items()}
        
        #extract actual ratio
        author_ratio = {hashtag: appearances/ len(authors) for hashtag, appearances in author_count.items()} if len(authors) > 0 else {hashtag: 0.0 for hashtag in author_count.keys()}

        return author_ratio

    def __getAuthor(self, tweetId):
        """
            returns the author of the specific tweet
        """
        return self.dbHandler.getTweetById(tweetId)["user"]["id_str"]

    def __get_tweet_ratio(self):
        """
            returns a dictionary of (hashtag, tweet ratio) attributes presenting the ratio of tweets containing the specific hashtag
        """
        tweet_count = {}
        #initialize dictionary
        for hashtag in self.hashtags:
            tweet_count[hashtag["text"]] = 0

        #count appearances
        for _, hashtag_list in self.tweet_hashtag_map.items():
            for hashtag in hashtag_list:
                tweet_count[hashtag["text"]] += 1
        
        tweet_ratio = {}
        #extract actual ratio
        tweet_ratio = {hashtag: appearances/ len(self.tweets) for hashtag, appearances in tweet_count.items()} if len(self.tweets) > 0 else {hashtag: 0.0 for hashtag in tweet_count.keys()}
        
        return tweet_ratio

    def __get_tweets_sentiment(self):
        """
            returns a dictionary of (tweet_id, positive/neutral/negative) attributes.
        """
        analyzer = SentimentIntensityAnalyzer()
        
        tweet_sentiment = {}
        for tweet in self.tweets:
            tweet_sentiment[tweet["id_str"]] = ""

        for tweet in self.tweets:
            text = self.__get_tweet_text(tweet)
            
            vs = analyzer.polarity_scores(text)
            sentiment = vs['compound']
            
            if sentiment >= 0.5:
                tweet_sentiment[tweet["id_str"]] = "positive"
            elif sentiment > -0.5 and sentiment < 0.5:
                tweet_sentiment[tweet["id_str"]] = "neutral"
            else:
                tweet_sentiment[tweet["id_str"]] = "negative"
        
        return tweet_sentiment

    def __get_hashtags_location(self):
        """
            finds the location of each hashtag inside the corresponding tweet text
            returns a dictionary of (hashtag, prefix/infix/postfix) attributes.
            The classification threshold is 0.25 of the total appearances
            !!!we do not check the case that tweet ends with more than 1 hashtags. The last one will be correctly classified as postfix but the others wrongly as infix!!!
        """
        prefix_counter = {}
        infix_counter = {}
        postfix_counter = {}
        for tweet_hashtag in self.tweet_hashtag_map.items():
            tweet_id = tweet_hashtag[0]
            tweet = self.dbHandler.getTweetById(tweet_id)
            
            text = self.__get_tweet_text(tweet)
           
            hashtag_list = tweet_hashtag[1]

            #initialize counters
            for hashtag in hashtag_list:
                prefix_counter[hashtag["text"]] = 0
                infix_counter[hashtag["text"]] = 0
                postfix_counter[hashtag["text"]] = 0

            for hashtag in hashtag_list:
                pattern = "#{}".format(hashtag["text"])

                #prefix
                if text.startswith(pattern):
                    prefix_counter[hashtag["text"]] += 1
                
                #infix
                if (not text.startswith(pattern)) and (not text.endswith(pattern)):#no need to check if hashtag exists in text as long as there is in tweet-hashtag map
                    infix_counter[hashtag["text"]] += 1
                
                #postfix
                if text.endswith(pattern):
                    postfix_counter[hashtag["text"]] += 1
            
        hashtag_location = {}
        for hashtag in self.hashtags:
            total_appearances = prefix_counter[hashtag["text"]] + infix_counter[hashtag["text"]] + postfix_counter[hashtag["text"]]
                
            prefix_ratio = float(prefix_counter[hashtag["text"]]) / total_appearances
            infix_ratio = float(infix_counter[hashtag["text"]]) / total_appearances
            postfix_ratio = float(postfix_counter[hashtag["text"]]) / total_appearances
                
            max_of_three = max(prefix_ratio, infix_ratio, postfix_ratio)

            if prefix_ratio == max_of_three and prefix_ratio > self.__LOCATION_THRESHOLD:
                hashtag_location[hashtag["text"]] = "prefix"
            elif infix_ratio == max_of_three and infix_ratio > self.__LOCATION_THRESHOLD:
                hashtag_location[hashtag["text"]] = "infix"
            else:
                hashtag_location[hashtag["text"]] = "postfix"
            
        return hashtag_location

    def __get_tweet_text(self, tweet):
        """
            returns the text of the given tweet json
        """
        if "extended_tweet" in tweet :
            text = tweet["extended_tweet"]["full_text"]
        elif "retweeted_status" in tweet and (not tweet["retweeted_status"]["truncated"]):
            text = tweet["retweeted_status"]["text"]
        elif "retweeted_status" in tweet and (tweet["retweeted_status"]["truncated"]):
            text = tweet["retweeted_status"]["extended_tweet"]["full_text"]
        else:
            text = tweet["text"]
        
        return str(text)

    def __get_hashtags_cooccurance(self):
        """
            returns a dictionary of (hashtag, true/false) attributes.
            True is given if more than 40% of the specific hashtag occurences are collocated with other hashtags
        """

        appearance_counter = {}
        cooccurance_counter = {} 
        
        #initialize counters
        for hashtag in self.hashtags:
            appearance_counter[hashtag["text"]] = 0
            cooccurance_counter[hashtag["text"]] = 0

        for hashtag_list in self.tweet_hashtag_map.values():
            if len(hashtag_list) == 1: #no coexisting hashtags in this list
                appearance_counter[hashtag_list[0]["text"]] += 1
            else:#only coexisting hashtags in this list
                for hashtag in hashtag_list:
                    appearance_counter[hashtag["text"]] += 1
                    cooccurance_counter[hashtag["text"]] += 1
            
        ###calculate ratio so as to consider 40% threshold
        ratio_hashtag_cooccurance = {}
        for hashtag, value in appearance_counter.items():
            ratio_hashtag_cooccurance[hashtag] = cooccurance_counter[hashtag] / float(value)

        for hashtag,value in ratio_hashtag_cooccurance.items():
            ratio_hashtag_cooccurance[hashtag] = True if ratio_hashtag_cooccurance[hashtag] >= self.__COOCCURANCE_THRESHOLD else False
            
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
        self.hashtags = hashtags

    def __get_hashtags_from_tweet(self, tweet, tweet_hashtag_map):
        """
            Private method used to extract hashtags from the given tweet.
        """
        hashtags = []
        #simple tweet
        if not tweet["truncated"]:
            first_level_potential_hashtags = tweet["entities"]["hashtags"]
            if first_level_potential_hashtags:
                hashtags.extend(first_level_potential_hashtags)

        #extended_tweet
        if tweet["truncated"] and "retweeted_status" not in tweet:
            extended_potential_hashtags = tweet["extended_tweet"]["entities"]["hashtags"]
            if extended_potential_hashtags:
                hashtags.extend(extended_potential_hashtags)
        
        #retweet with no extended_tweet field
        if "retweeted_status" in tweet and (not tweet["retweeted_status"]["truncated"]):
            retweeted_potential_hashtags = tweet["retweeted_status"]["entities"]["hashtags"]
            if retweeted_potential_hashtags:
                hashtags.extend(retweeted_potential_hashtags)
       
        #retweet with extended_tweet field
        if ("retweeted_status" in tweet) and (tweet["retweeted_status"]["truncated"]):
            retweeted_potential_hashtags = tweet["retweeted_status"]["extended_tweet"]["entities"]["hashtags"]
            if retweeted_potential_hashtags:
                hashtags.extend(retweeted_potential_hashtags)

        tweet_hashtag_map[tweet["id_str"]] = hashtags

        
        
                