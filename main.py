import DbHandler
import FeatureExtractor

if __name__ == '__main__':
    db_handler = DbHandler.DbHandler()

    feature_extractor = FeatureExtractor.FeatureExtractor(db_handler)

    hashtag_features = feature_extractor.get_hashtag_features()
    tweet_features = feature_extractor.get_tweet_features()

    #print(hashtag_features)
    print(tweet_features)