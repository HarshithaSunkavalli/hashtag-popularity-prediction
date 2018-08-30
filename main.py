import DbHandler

db_handler = DbHandler.DbHandler()

tweets = db_handler.getTweets()

for tweet in tweets:
    print(tweet)