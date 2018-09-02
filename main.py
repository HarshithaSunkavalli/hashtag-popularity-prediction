import DbHandler
import FeatureExtractor

db_handler = DbHandler.DbHandler()
tweets = db_handler.tweets

feature_extractor = FeatureExtractor.FeatureExtractor(tweets, db_handler)

hashtag_features = feature_extractor.get_hashtag_features()
tweet_features = feature_extractor.get_tweet_features()

print(hashtag_features)
print(tweet_features)