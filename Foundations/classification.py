
import pandas as pd
import numpy as np
from foundations import Tokenize
from embeddings import VectorEmbeddings
from sklearn.linear_model import LogisticRegression


class Classification():
    def __init__(
        self,
        documents:pd.Series,
        doc_label:pd.Series,
        training_documents:int,
        split:int=0.8
    ):
        self.documents = documents
        self.train_docs = training_documents
        self.doc_labels = doc_label
        self.split = split

        self.doc_vectors = np.zeros((self.train_docs, 100))
        self.labels = np.zeros(self.train_docs)

        self.document_vectorizer = VectorEmbeddings(
            self.documents,
            100,
            self.train_docs,
            num_docs=self.train_docs
        )

    def _prepare_data(self):
        self.document_vectorizer.fit_model()
        for i in range(self.train_docs):
            self.doc_vectors[i] = self.document_vectorizer.infer_vector(
                self.documents,
                i+1
            )

            if self.doc_labels[i] == 'positive': self.labels[i] = 1
            else: self.labels[i] = 0

    def train_classifier(self):
        self._prepare_data()
        self.model = LogisticRegression(random_state=42).fit(
            self.doc_vectors,
            self.labels
        )

    def predict(self, doc_index:int):
        doc_vect = self.document_vectorizer.infer_vector(self.documents, doc_index)
        return self.model.predict(
            doc_vect.reshape(1, -1)
        )



if __name__ == '__main__':
    data = pd.read_csv('data.csv')['review']
    labels = pd.read_csv('data.csv')['sentiment']

    model = Classification(data, labels, 200)
    model.train_classifier()

    print(model.predict(389))
    print(labels[389])