from collections import Counter
from collections import OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib as mpl

class PlotFactory:

    K = 10

    def __init__(self, dbHandler, data):
        self.dbHandler = dbHandler
        self.data = data
        self.hashtags = self.data["hashtag"].values

        self.get_tweets_for_hashtags()

        mpl.style.use("seaborn")

    def get_tweets_for_hashtags(self):
        """
        :return: a dictionary with hashtags as key and a list of all the tweets contain the key hashtag as value.
        """

        self.hashtag_tweet_map = {}
        for hashtag in self.hashtags:
            self.hashtag_tweet_map[hashtag] = []

        for hashtag in self.hashtags:
            tweets = self.dbHandler.getTweetsForHashtag(hashtag)
            self.hashtag_tweet_map[hashtag].extend(tweets)

    def hashtag_appearance(self, hashtag="India"):
        """
        Calculates frequency by date for a given hashtag and plots.
        """
        tweets = self.hashtag_tweet_map[hashtag]
        appearances = [tweet["created_at"] for tweet in tweets]
        #keep only date and throw time
        appearances = [datetime.date() for datetime in appearances]

        date_frequency = Counter(appearances) #key: date, value: frequency

        plt.plot(date_frequency.keys(), date_frequency.values(), 'go-')
        plt.title("Hashtag:{} Frequency by Date".format(hashtag))
        plt.xlabel("Date")
        plt.ylabel("Frequency")
        plt.show()

    def hashtag_appearance_for_top_k(self):
        """
        Calculates the top K hashtags and plots their date-frequency plot
        """

        top_k_popular_hashtags = self.data.sort_values(by='popularity', ascending=False).head(self.K)
        for hashtag in top_k_popular_hashtags["hashtag"]:
            self.hashtag_appearance(hashtag)

