from collections import Counter
from math import log


class CountVectorizer:
    """
    Count vectorizer class
    """

    def __init__(self):
        self.vocab = set()

    def fit_transform(self, texts: list) -> list:
        """
        :param texts: list of strings
        :return: list of list
        """
        for i in range(len(texts)):
            texts[i] = texts[i].lower()
            self.vocab = self.vocab.union(set(texts[i].split(" ")))
        vecs = []
        for text in texts:
            vec_map = {token: 0 for token in self.vocab}
            counter = dict(Counter(text.split(" ")))
            vec_map.update(counter)
            vecs.append(list(vec_map.values()))
        return vecs

    def get_names(self) -> list:
        """
        :return: list of strings
        """
        return list(self.vocab)


class TfidfTransformer:
    def fit_transform(self, count_matrix: list) -> list:
        """
        :param count_matrix: list of list[int]
        :return: list of list[float]
        """
        tf = self.tf_transform(count_matrix)
        idf = self.idf_transform(count_matrix)
        sh0 = len(tf)
        if sh0 == 0:
            return []
        sh1 = len(tf[0])
        return [[tf[i][j] * idf[j] for j in range(sh1)] for i in range(sh0)]
    
    def tf_transform(self, count_matrix: list) -> list:
        """
        :param count_matrix: list of list[int]
        :return: list of list[float]
        """
        return [
            [count_row[i] / sum(count_row) for i in range(len(count_row))]
            for count_row in count_matrix
        ]


    def idf_transform(self, count_matrix: list) -> list:
        """
        :param count_matrix: list of list[int]
        :return: list of float
        """
        sh0 = len(count_matrix)
        if sh0 == 0:
            return []
        sh1 = len(count_matrix[0])
        words_docs = [
            sum([count_matrix[i][j] > 0 for i in range(sh0)]) for j in range(sh1)
        ]
        return [log((sh0 + 1) / (words_docs[i] + 1)) + 1 for i in range(sh1)]


class TfidfVectorizer(CountVectorizer):
    def __init__(self):
        super(TfidfVectorizer, self).__init__()
        self.tf_idf_transformer = TfidfTransformer()

    def fit_transform(self, texts: list) -> list:
        """
        :param texts: list of strings
        :return: list of list[float]
        """
        count_matrix = super().fit_transform(texts)
        return self.tf_idf_transformer.fit_transform(count_matrix)

    def get_names(self) -> list:
        """
        :return: list of strings
        """
        return super().get_names()


if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_names())
    print(tfidf_matrix)
