import DbHandler
import PlotFactory
from FeatureExtractors.HashtagFeatureExtractor import HashtagFeatureExtractor
from FeatureExtractors.TweetFeatureExtractor import TweetFeatureExtractor
from FeatureExtractors.IOHandler import IOHandler
from Predictors.DBScan import DBScan
from Predictors.GRUNN import GRUNN
from Predictors.NaiveBayes import NaiveBayes
from Predictors.KNN import KNN
from Predictors.DecisionTree import DecisionTree
from Predictors.SVM import SVM
from Predictors.LR import LR
from Predictors.RandomPredictor import RandomPredictor
from Predictors.PriorDist import PriorDist

CLUSTERING = "SVM"

def createFeatureCSV(db_handler):
    """
        Processes tweets and hashtags to produce the necessary features.
        Writes the features in a CSV.
    """


    feature_extractor = HashtagFeatureExtractor(db_handler, k=True)
    tweet_feature_extractor = TweetFeatureExtractor(db_handler, k=True)

    hashtag_features = feature_extractor.get_hashtag_features()
    tweet_features = tweet_feature_extractor.get_tweet_features()
    features = hashtag_features
    features.update(tweet_features)

    ioHandler = IOHandler()
    data, labels = ioHandler.preprocessHashtagFeatures(hashtag_features)

    ioHandler.writeToCSV(labels, data)


if __name__ == '__main__':
    db_handler = DbHandler.DbHandler()
    #createFeatureCSV(db_handler)
    ioHandler = IOHandler()
    data = ioHandler.readFromCSV("top_k.csv")
    ioHandler.top_k_hashtags_CSV(data, 10)

    # plot_factory = PlotFactory.PlotFactory(db_handler, data)
    # plot_factory.hashtag_appearance_for_top_k()

    if CLUSTERING == "DBSCAN":
        dbscan = DBScan(users=data, eps=0.3, MinPts=5, reduce_dimensions=True)
        labels, NumClusters = dbscan.run()
        data.loc[:, "label"] = labels
    elif CLUSTERING == "GRUNN":
        train_data = ioHandler.readFromCSV() # read from features.csv
        grunn = GRUNN(train=train_data, test=data, reduce_dimensions=True)
        labels = grunn.train()
    elif CLUSTERING == "NaiveBayes":
        train_data = ioHandler.readFromCSV()
        nB = NaiveBayes(train=train_data, test=data, reduce_dimensions=True)
        labels = nB.run()
        data.loc[:, "label"] = labels
    elif CLUSTERING == "KNN":
        train_data = ioHandler.readFromCSV()
        knn =  KNN(train=train_data, test=data, reduce_dimensions=True)
        labels = knn.run(k=3)
        data.loc[:, "label"] = labels
    elif CLUSTERING =="DecisionTree":
        train_data = ioHandler.readFromCSV()
        dTree = DecisionTree(train=train_data, test=data, reduce_dimensions=True)
        labels = dTree.run()
        data.loc[:, "label"] = labels
    elif CLUSTERING =="SVM":
        train_data = ioHandler.readFromCSV()
        svm = SVM(train=train_data, test=data, reduce_dimensions=False)
        labels = svm.run()
        data.loc[:, "label"] = labels
    elif CLUSTERING =="LR":
        train_data = ioHandler.readFromCSV()
        lr = LR(train=train_data, test=data, reduce_dimensions=False)
        labels = lr.run()
        data.loc[:, "label"] = labels
    elif CLUSTERING =="Random":
        train_data = ioHandler.readFromCSV()
        rp = RandomPredictor(train=train_data, test=data, reduce_dimensions=False)
        labels = rp.run()
        data.loc[:, "label"] = labels
    elif CLUSTERING =="PriorDist":
        train_data = ioHandler.readFromCSV()
        pd = PriorDist(train=train_data, test=data, reduce_dimensions=False)
        labels = pd.run()
        data.loc[:, "label"] = labels
    else:
        pass