
class GRUNN:

    def __init__(self, data):
        self.data = data

    def extractLabels(self):
        threshold = 10000
        labels = [0 if row["popularity"] < threshold else 1 for _, row in self.data.iterrows() ]

        self.data["label"] = labels

    def run(self):

        self.extractLabels()
        print(self.data.columns)
