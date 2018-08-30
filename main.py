import DbHandler
import FeatureExtractor

db_handler = DbHandler.DbHandler()

tweets = db_handler.getTweets()

feature_extractor = FeatureExtractor.FeatureExtractor(tweets)
feature_extractor.get_hashtag_features()
