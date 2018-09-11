import csv

class IOHandler:

    def preprocessHashtagFeatures(self, data):
        """
        :param data: data in the form of feature extractors.
        :return: list of dictionary key value items. First key: value must be hashtag: hashtag value.
        """

        labels = ["hashtag"] #first column should be the hashtag itself
        hashtags = []
        values = {} # {feature1: [hash1 val, hash2 val, ...], feature2: ...}

        hashtags_completed = False #needed because hashtags are repeated
        for feature, hashtagsAndValues in data.items():
            values[feature] = []
            labels.append(feature)
            for hashtag, value in hashtagsAndValues.items():
                if not hashtags_completed:
                    hashtags.append(hashtag)

                values[feature].append(value)
            hashtags_completed = True

        featureValue = []
        #labels length == hashtags length
        for i in range(len(hashtags)):
            dictionary = {}
            dictionary["hashtag"] = hashtags[i] # first pair is the hashtag itself
            for index, feature in enumerate(values.keys()):
                feature_values = list(values.values())[index]
                dictionary[feature] = feature_values[i] # for feature in position index take value for corresponding hashtag
            featureValue.append(dictionary)

        return featureValue, labels

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
