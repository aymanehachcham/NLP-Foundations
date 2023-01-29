
import pandas as pd
import os
import random
import string
from Foundations.foundations import Tokenizer
import unittest
from unittest import TestCase


DATA_ROOT_DIR = '../data'
FILE_NAME = 'hamlet.txt'
SERIES = 'data.csv'

class TestTokenizer(TestCase):

    def test_init_with_text_file(self):
        tokenizer = Tokenizer(
            root_dir=DATA_ROOT_DIR,
            file_name=FILE_NAME
        )
        self.assertEqual(tokenizer.root_dir, DATA_ROOT_DIR)
        self.assertEqual(tokenizer.filename, FILE_NAME)
        self.assertFalse(tokenizer.stopwords)
        self.assertFalse(tokenizer.normalized)
        self.assertIsNone(tokenizer._df)


    def test_init_with_series(self):
        tokenizer = Tokenizer(
            series=pd.read_csv(os.path.join(DATA_ROOT_DIR, SERIES))['review']
        )
        self.assertIsInstance(tokenizer._df, pd.Series)
        self.assertIsNotNone(tokenizer._df)
        self.assertIsNotNone(tokenizer.list_docs)
        self.assertIsInstance(tokenizer.list_docs, list)
        self.assertEqual(tokenizer.root_dir, './')
        self.assertEqual(tokenizer.filename, 'file.txt')

    def test_split_into_tokens_flatten(self):
        tokenizer = Tokenizer(
            series=pd.read_csv(os.path.join(DATA_ROOT_DIR, SERIES))['review']
        )
        text_to_split = tokenizer.list_docs[random.randint(0, 100)]
        tokens = tokenizer._split_into_tokens(
            text=text_to_split,
            flatten=True
        )
        self.assertIsNotNone(tokens)
        self.assertIsInstance(tokens, list)
        self.assertNotIsInstance(tokens[0], list)
        self.assertTrue([token != '' for token in tokens])
        self.assertTrue([token.islower() != '' for token in tokens])
        self.assertTrue([len(item) > 1 for item in tokens])
        self.assertTrue(len(list(filter(lambda x: x in string.punctuation, tokens))) == 0)
        self.assertIsNotNone(tokenizer._tokens)

    def test_split_into_tokens_not_flatten(self):
        tokenizer = Tokenizer(
            series=pd.read_csv(os.path.join(DATA_ROOT_DIR, SERIES))['review']
        )
        text_to_split = tokenizer.list_docs[random.randint(0, 100)]
        tokens = tokenizer._split_into_tokens(
            text=text_to_split,
            flatten=False
        )
        self.assertIsNotNone(tokens)
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(tokens[0], list)
        self.assertIsNotNone(tokenizer._tokens)


if __name__ == '__main__':
    unittest.main()


