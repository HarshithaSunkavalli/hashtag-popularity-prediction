from pymongo import MongoClient

class DbHandler:

    def __init__(self):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client["Test_db"]

    def getTweets(self):
        tweets = self.db["plastic"].find({}) #use curly brackets to bypass default return limit
        return tweets

    def getTweetsByNum(self, num):
        tweets = self.db["plastic"].find({}).limit(num)
        return tweets

    def getTweetById(self, id):
        tweet = self.db["plastic"].find_one({"id_str": id})
        return tweet