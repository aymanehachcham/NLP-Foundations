import numpy as np

from foundations import Tokenize
import unittest
from unittest import TestCase


## Start creating class as tests
tokenizer = Tokenize(
    root_dir='../data',
    file_name='hamlet.txt',
    stopwords=True,
    normalized=True
)
tokenizer.most_common_tokens(5)
tokens = tokenizer.get_tokens()


if __name__ == '__main__':
    unittest.main()


