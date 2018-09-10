import DbHandler
from FeatureExtractors.HashtagFeatureExtractor import HashtagFeatureExtractor
from FeatureExtractors.TweetFeatureExtractor import TweetFeatureExtractor
from FeatureExtractors.IOHandler import IOHandler

if __name__ == '__main__':
    db_handler = DbHandler.DbHandler()

    feature_extractor = HashtagFeatureExtractor(db_handler)
    tweet_feature_extractor = TweetFeatureExtractor(db_handler)

    hashtag_features = feature_extractor.get_hashtag_features()
    tweet_features = tweet_feature_extractor.get_tweet_features()

    print(hashtag_features)
    print(tweet_features)

    ioHandler = IOHandler()

    data = [{"hashtag":"PGP", "Words": 1, "Caps": 1}, {"hashtag":"HelloThere", "Words": 2, "Caps": 0}]
    labels = ["hashtag","Words","Caps"]
    ioHandler.writeToCSV(labels, data)
