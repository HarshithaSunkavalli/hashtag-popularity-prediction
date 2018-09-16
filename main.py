import DbHandler
from FeatureExtractors.HashtagFeatureExtractor import HashtagFeatureExtractor
from FeatureExtractors.TweetFeatureExtractor import TweetFeatureExtractor
from FeatureExtractors.IOHandler import IOHandler
from DeepLearning.AutoEncoder import AutoEncoder

if __name__ == '__main__':
    db_handler = DbHandler.DbHandler()

    feature_extractor = HashtagFeatureExtractor(db_handler, k=True)
    tweet_feature_extractor = TweetFeatureExtractor(db_handler, k=True)

    hashtag_features = feature_extractor.get_hashtag_features()
    tweet_features = tweet_feature_extractor.get_tweet_features()

    features = hashtag_features
    features.update(tweet_features)

    ioHandler = IOHandler()
    data, labels = ioHandler.preprocessHashtagFeatures(hashtag_features)

    ioHandler.writeToCSV(labels, data)

    autoencoder = AutoEncoder()
    autoencoder.reduce_dimensions()