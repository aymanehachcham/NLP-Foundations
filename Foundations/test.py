import numpy as np

from foundations import Tokenize
import unittest


## Start creating class as tests
tokenizer = Tokenize(
    root_dir='../data',
    file_name='hamlet.txt',
    stopwords=True,
    normalized=True
)
tokenizer.most_common_tokens(5)
tokens = tokenizer.get_tokens()

############# Tests:
# assert not '' in tokens,  'Empty tokens are not suitable'
# # assert not any([len(token) == 1 for token in tokens]), 'One character tokens are not allowed'
#
for i in np.argwhere(np.array([len(token) == 1 for token in tokens])).flatten():
    print(len(tokens[i]))


