from tqdm import tqdm
import DbHandler
import PlotFactory
from FeatureExtractors.FeatureExtractor import FeatureExtractor
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
FEATURE_EXTRACTION = False
CREATE_CSV = False

def createFeatureCSV(db_handler, ioHandler):
    """
        Processes tweets and hashtags to produce the necessary features.
        Writes the features in a CSV.
    """
    feature_extractor = FeatureExtractor(db_handler)

    if CREATE_CSV:
        print("Extracting hashtags from tweets")
        #feature_extractor.create_hashtag_csv(ioHandler)
        feature_extractor.create_top_k_csv(ioHandler)
    else:
        hashtag_feature_extractor = HashtagFeatureExtractor(featureExtractor=feature_extractor)
        tweet_feature_extractor = TweetFeatureExtractor(featureExtractor=feature_extractor)
        tweet_feature_extractor.precalculateValues()

        hashtags = ioHandler.readFromCSV("hashtags.csv")["hashtag"]

        for index, hashtag in tqdm(enumerate(hashtags)):
            features = {}
            print("Hashtag: ", hashtag)
            features.update({"hashtag": hashtag})

            hashtag_features = hashtag_feature_extractor.get_hashtag_features(hashtag)
            features.update(hashtag_features)

            tweet_features = tweet_feature_extractor.get_tweet_features(hashtag)
            features.update(tweet_features)

            if index == 0:
                header = True
            else:
                header = False

            ioHandler.writeToCSV(features, header)


if __name__ == '__main__':
    db_handler = DbHandler.DbHandler()
    ioHandler = IOHandler()
    if FEATURE_EXTRACTION:
        createFeatureCSV(db_handler, ioHandler)
    else:
        top_hashtags = ioHandler.readFromCSV("top_k.csv")

        data = ioHandler.readFromCSV("features_related.csv")

        test_data = data.loc[data["hashtag"].isin(top_hashtags["hashtag"])]
        test_data = test_data.reset_index(drop=True)

        train_data = data.loc[~data["hashtag"].isin(top_hashtags["hashtag"])]
        train_data = train_data.reset_index(drop=True)


        # plot_factory = PlotFactory.PlotFactory(db_handler, data)
        # plot_factory.hashtag_appearance_for_top_k()
        print("Clustering")
        if CLUSTERING == "DBSCAN":
            dbscan = DBScan(users=train_data, eps=0.3, MinPts=5, reduce_dimensions=True)
            labels, NumClusters = dbscan.run()
            test_data.loc[:, "label"] = labels
        elif CLUSTERING == "GRUNN":
            grunn = GRUNN(train=train_data, test=test_data, reduce_dimensions=True)
            labels = grunn.train()
        elif CLUSTERING == "NaiveBayes":
            nB = NaiveBayes(train=train_data, test=test_data, reduce_dimensions=True)
            labels = nB.run()
            test_data.loc[:, "label"] = labels
        elif CLUSTERING == "KNN":
            knn =  KNN(train=train_data, test=test_data, reduce_dimensions=True)
            labels = knn.run(k=3)
            test_data.loc[:, "label"] = labels
        elif CLUSTERING =="DecisionTree":
            dTree = DecisionTree(train=train_data, test=test_data, reduce_dimensions=True)
            labels = dTree.run()
            test_data.loc[:, "label"] = labels
        elif CLUSTERING =="SVM":
            svm = SVM(train=train_data, test=test_data, reduce_dimensions=True)
            labels = svm.run()
            test_data.loc[:, "label"] = labels
        elif CLUSTERING =="LR":
            lr = LR(train=train_data, test=test_data, reduce_dimensions=True)
            labels = lr.run()
            test_data.loc[:, "label"] = labels
        elif CLUSTERING =="Random":
            rp = RandomPredictor(train=train_data, test=test_data, reduce_dimensions=False)
            labels = rp.run()
            test_data.loc[:, "label"] = labels
        elif CLUSTERING =="PriorDist":
            pd = PriorDist(train=train_data, test=test_data, reduce_dimensions=False)
            labels = pd.run()
            test_data.loc[:, "label"] = labels
        else:
            pass