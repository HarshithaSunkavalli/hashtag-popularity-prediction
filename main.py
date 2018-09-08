import DbHandler
from FeatureExtractors.HashtagFeatureExtractor import HashtagFeatureExtractor
from FeatureExtractors.TweetFeatureExtractor import TweetFeatureExtractor

if __name__ == '__main__':
    db_handler = DbHandler.DbHandler()

    feature_extractor = HashtagFeatureExtractor(db_handler)
    tweet_feature_extractor = TweetFeatureExtractor(db_handler)

    hashtag_features = feature_extractor.get_hashtag_features()
    tweet_features = tweet_feature_extractor.get_tweet_features()

    print(hashtag_features)
    print(tweet_features)