from pymongo import MongoClient

class DbHandler:

    def __init__(self):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client["Test_db"]

    def getTweets(self):
        tweets = self.db["tweets"].find({}) #use curly brackets to bypass default return limit
        return tweets

