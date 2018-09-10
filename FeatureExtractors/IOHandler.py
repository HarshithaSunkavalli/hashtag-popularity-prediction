import csv

class IOHandler:


    def writeToCSV(self, labels, data):
        """
        :param labels: the csv column names
        :param data: list of dictionary key value items. First key: value must be hashtag: hashtag value.
        """

        with open("features.csv", "w", newline="") as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=labels)
            writer.writeheader()

            for i in range(len(data)):
                dictionary = {} # contains a specific hashtag features
                for label, value in data[i].items():
                    dictionary.update({label: value})

                writer.writerow(dictionary)
