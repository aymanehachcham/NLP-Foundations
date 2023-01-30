
import unittest
import pandas as pd
import os

from Foundations.embeddings import VectorEmbeddings

DATA_ROOT_DIR = '../../data'
FILE_NAME = 'hamlet.txt'
SERIES = 'data.csv'


class TestDoc2VecEmbedding(unittest.TestCase):

    def test_doc2vec_init(self):
        embedding = VectorEmbeddings(
            df=pd.read_csv(os.path.join(DATA_ROOT_DIR, SERIES))['review'],
            embedding='doc2vec',
        )

        # doc = pd.read_csv(os.path.join(DATA_ROOT_DIR, SERIES))['review'][300]
        # embedding.load_model()
        # print(embedding.infer_vector(doc))


if __name__ == '__main__':
    unittest.main()