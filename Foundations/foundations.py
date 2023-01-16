
import os
import re
import nltk
from typing import List

class Tokenize():
    """
    This class tokenizes any given text in file format.

    Parameters
    ----------
    root_dir: str,
    The root directory of the folder containing the file wit the text

    file_name: str,
    The name of the file containing the text data

    stopwords: bool,
    Include stopword removal in the tokenization

    normalized: bool,
    Apply normalization on the tokens -> Lemmatization
    """
    def __init__(
        self,
        root_dir:str,
        file_name:str,
        stopwords:bool=False,
        normalized:bool=False
    ):

        self._sentence_pattern = re.compile("[.:,!;\n]\s", re.U)
        self.stopwords = stopwords
        self.normalized = normalized
        self.tokens_filled = False
        self._tokens = []

        if not os.path.exists(os.path.join(root_dir, file_name)):
            raise ValueError(
                'The given path: {} does not exist'.format(os.path.join(root_dir, file_name))
            )
        else:
            file = os.path.join(root_dir, file_name)
            with open(file) as f: self.full_text = f.read()


    @staticmethod
    def _preprocess_text(text:str) -> str:
        space_pattern = '\s+'
        new_line = '\n+'
        mention_regex = '@[\w\-]+'
        non_word_char = '[^\w]'
        underscore = '_[\w]+'

        parsed_text = re.sub(space_pattern, ' ', text)
        parsed_text = re.sub(new_line, ' ', parsed_text)
        parsed_text = re.sub(mention_regex, '', parsed_text)
        parsed_text = re.sub(non_word_char, ' ', parsed_text)
        parsed_text = re.sub(r"\bو(.*?)\b", r'\1', parsed_text)
        parsed_text = re.sub('([0-9]+)', '', parsed_text)
        parsed_text = re.sub(underscore, ' ', parsed_text)

        return parsed_text


    def _split_into_tokens(self, text:str):
        """
        Initiate the splitting process and updates the tokens list
        """
        processed_text = self._preprocess_text(text)

        for line in re.split(self._sentence_pattern, processed_text):
            if line.strip() == '':
                continue
            else:
                line = line.lower()
                self._tokens.append(line.split())

        # Flatten the tokens
        self._tokens = [token for item in self._tokens for token in item]

    def get_tokens(self):
        """
        Returns the list of all tokens after applying stopword removal and normalization
        """
        self._split_into_tokens(self.full_text)

        if self.stopwords:
            from nltk.corpus import stopwords
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')

            _stop_words = set(stopwords.words('english'))

            # Remove stopwords:
            self._tokens = list(filter(lambda x: not x in _stop_words, self._tokens))

        if self.normalized:
            from nltk.stem import WordNetLemmatizer
            try:
                nltk.data.find('corpora/wordnet.zip')
            except LookupError:
                nltk.download('wordnet')

            lematizer = WordNetLemmatizer()

            # Lemmatize all the tokens:
            normalized_tokens = list(map(lambda x: lematizer.lemmatize(x, pos='v'), self._tokens))
            normalized_tokens = list(map(lambda x: lematizer.lemmatize(x, pos='n'), normalized_tokens))
            normalized_tokens = list(map(lambda x: lematizer.lemmatize(x, pos='a'), normalized_tokens))
            self._tokens = normalized_tokens

            # Remove all unitary tokens
            self._tokens = list(filter(lambda x: len(x) > 1, self._tokens))
            self.tokens_filled = True

        return self._tokens

    def _check_tokens(self):
        if not self.tokens_filled:
            raise ValueError(
                'There is no tokens to process'
            )

    def most_common_tokens(self, num:int) -> list:
        self._check_tokens()

        from collections import Counter
        counter = Counter(self._tokens)
        return counter.most_common(num)
