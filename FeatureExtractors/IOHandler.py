import csv
import pandas as pd


class IOHandler:

    def writeToCSV(self, data, header, my_csv="features.csv"):

        labels = list(data.keys())

        with open(my_csv, "a", newline="", encoding="utf-8") as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=labels)

            if header:
                writer.writeheader()

            writer.writerow(data)

    def writeListToCSV(self, l, label="hashtag", my_csv="hashtags.csv"):

        with open(my_csv, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[label])
            writer.writeheader()

            for value in l:
                writer.writerow({label: value})


    def readFromCSV(self, csv="features.csv"):
        return pd.read_csv(csv)

    def top_k_hashtags_CSV(self, data, k):
        top_k_popular_hashtags = data.sort_values(by='popularity', ascending=False).head(k)
        top_k_popular_hashtags.to_csv("top_k.csv", encoding='utf-8', index=False)
