import DbHandler
import FeatureExtractor

db_handler = DbHandler.DbHandler()
tweets = db_handler.getTweets()

feature_extractor = FeatureExtractor.FeatureExtractor(tweets)

hashtag_features = feature_extractor.get_hashtag_features()

print(hashtag_features)
