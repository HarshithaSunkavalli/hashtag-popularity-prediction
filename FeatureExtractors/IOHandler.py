import csv
import pandas as pd


class IOHandler:

    def writeToCSV(self, data, header):
        """
        :param labels: the csv column names
        :param data: list of dictionary key value items. First key: value must be hashtag: hashtag value.
        """

        labels = list(data.keys())

        with open("features.csv", "a", newline="", encoding="utf-8") as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=labels)

            if header:
                writer.writeheader()

            writer.writerow(data)

    def readFromCSV(self, csv="features.csv"):
        return pd.read_csv(csv)

    def top_k_hashtags_CSV(self, data, k):
        top_k_popular_hashtags = data.sort_values(by='popularity', ascending=False).head(k)
        top_k_popular_hashtags.to_csv("top_k.csv", encoding='utf-8', index=False)
