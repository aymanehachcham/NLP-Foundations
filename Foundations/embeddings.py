import pandas as pd
import torch

from foundations import Tokenize
import gensim
from gensim.models.doc2vec import TaggedDocument
from typing import Literal


class VectorEmbeddings():
    def __init__(
        self,
        df:pd.Series,
        doc_vect_size:int=100,
        min_freq_word_count:int=2,
        num_docs:int=100,
        embedding:Literal['doc2vec', 'bert']='doc2vec'
    ):
        self._tokens = []
        self.model = None
        self.df = df
        self.min_freq_word_count = min_freq_word_count
        self.vocab:bool=False
        self.num_docs = num_docs

        if embedding in ['doc2vec', 'bert']:
            self.embedding = embedding
        else:
            raise ValueError(
                'The embedding mode is incorrect'
            )

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


    def _bert_case_preparation(self):
        from transformers import BertTokenizer, BertModel
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(
            'bert-base-uncased',
            output_hidden_states = True,
        )
        self.model.eval()
        self.vocab = True


    def _preprocess_documents(self):
        self.tokens = self.num_docs
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

    def _fit_doc2vec_model(self):
        self._build_vocab()
        self.model.train(
            self.train_corpus,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs
        )
        self.vocab = True

    def load_model(self):
        if self.embedding == 'bert':
            self._bert_case_preparation()
        if self.embedding == 'doc2vec':
            self._fit_doc2vec_model()

    def infer_vector(self, doc:pd.Series, index:int):
        if self.vocab:
            if self.embedding == 'doc2vec':
                doc_tokenized = Tokenize(
                    series=doc[index - 1:index],
                    stopwords=True,
                    normalized=True
                ).get_tokens()[0]

                return self.model.infer_vector(doc_tokenized)

            if self.embedding == 'bert':
                sentence_inputs = list(filter(lambda x:x!='', doc[index - 1:index].tolist()[0].split('.')))
                doc_input = ''.join(sentence_inputs[:10])

                marked_text = "[CLS] " + doc_input + " [SEP]"
                tokens = self.bert_tokenizer.tokenize(marked_text)
                idx = self.bert_tokenizer.convert_tokens_to_ids(tokens)
                segment_id = [1] * len(tokens)

                self.tokens_tensor = torch.tensor([idx])
                self.segments_tensors = torch.tensor([segment_id])

                with torch.no_grad():
                    outputs = self.model(self.tokens_tensor, self.segments_tensors)
                    hidden_states = outputs[2]

                self.hidden_states = hidden_states

                return torch.mean(self.hidden_states[-2][0], dim=0)
        else:
            raise ValueError(
                'The Embedding model has not been initialized'
            )




df = pd.read_csv('data.csv')['review']
doc = df[10:11]

token_model = VectorEmbeddings(df, embedding='bert')
token_model.load_model()
print(token_model.infer_vector(df, 100))












