import pandas as pd
from foundations import Tokenize
import gensim
from gensim.models.doc2vec import TaggedDocument

class VectorEmbeddings():
    def __init__(
        self,
        df:pd.Series,
        doc_vect_size:int,
        min_freq_word_count:int=2,
    ):
        self._tokens = []
        self.model = None
        self.df = df
        self.min_freq_word_count = min_freq_word_count
        self.vocab:bool=False

        if doc_vect_size < 100:
            self.doc_vect_size = 100
        else:
            self.doc_vect_size = doc_vect_size

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, docs:int):
        self._tokens = Tokenize(
            series=self.df[:docs],
            stopwords=True,
            normalized=True
        ).get_tokens()

    def _preprocess_documents(self):
        self.tokens = 100
        for i, doc in enumerate(self._tokens):
            yield gensim.models.doc2vec.TaggedDocument(doc, [i])

    def _build_vocab(self):
        self.train_corpus = list(self._preprocess_documents())
        self.model =  gensim.models.doc2vec.Doc2Vec(
            vector_size=self.doc_vect_size,
            min_count=self.min_freq_word_count,
            epochs=40
        )
        self.model.build_vocab(self.train_corpus)
        self.vocab = True

    def infer_vector(self, doc:pd.Series, index:int):
        self._build_vocab()
        self.model.train(
            self.train_corpus,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs
        )

        doc = Tokenize(
            series=doc[index-1:index],
            stopwords=True,
            normalized=True
        ).get_tokens()[0]

        return self.model.infer_vector(doc)
