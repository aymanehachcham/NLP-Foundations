
import os
import re
import nltk
import pandas as pd

class Tokenizer():
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

    CURRENT_PATH = './'
    EMPTY_FILE = 'file.txt'

    def __init__(
        self,
        root_dir:str=CURRENT_PATH,
        file_name:str=EMPTY_FILE,
        series:pd.Series=None,
        stopwords:bool=False,
        normalized:bool=False
    ):

        self._sentence_pattern = re.compile("[.:,!;\n]\s", re.U)
        self.root_dir = root_dir
        self.filename = file_name
        self.stopwords = stopwords
        self.normalized = normalized
        self.tokens_filled = False
        self._tokens = []
        self._df = series

        if self.stopwords:
            ## Stop words loading
            from nltk.corpus import stopwords
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')

            self.stop_words = set(stopwords.words('english'))

        if self.normalized:
            ## Lemmatizer loading
            from nltk.stem import WordNetLemmatizer
            try:
                nltk.data.find('corpora/wordnet.zip')
            except LookupError:
                nltk.download('wordnet')

            self.lematizer = WordNetLemmatizer()


        if not os.path.exists(os.path.join(root_dir, file_name)) and self._df is None:
            raise ValueError(
                'Tokenizer must be initialized with either a pandas Series or a text file'
            )

        elif self._df is not None:
            try:
                self.list_docs = self._df.tolist()
            except Exception:
                raise ValueError(
                    'The given Series is not valid'
                )

        elif not os.path.exists(os.path.join(root_dir, file_name)):
            raise ValueError(
                'The given path: {} does not exist'.format(os.path.join(root_dir, file_name))
            )
        else:
            file = os.path.join(root_dir, file_name)
            with open(file) as f: self.full_text = f.read()

    # def trackcalls(self, func):
    #     def wrapper(*args, **kwargs):
    #         wrapper.has_been_called = True
    #         return func(*args, **kwargs)
    #
    #     wrapper.has_been_called = False
    #     return wrapper

    @staticmethod
    def _preprocess_text(text:str) -> str:
        space_pattern = '\s+'
        new_line = '\n+'
        mention_regex = '@[\w\-]+'
        non_word_char = '[^\w]'
        underscore = '_[\w]+'
        html_tags = '<.*?>'

        parsed_text = re.sub(space_pattern, ' ', text)
        parsed_text = re.sub(html_tags, ' ', parsed_text)
        parsed_text = re.sub(new_line, ' ', parsed_text)
        parsed_text = re.sub(mention_regex, '', parsed_text)
        parsed_text = re.sub(non_word_char, ' ', parsed_text)
        parsed_text = re.sub(r"\bو(.*?)\b", r'\1', parsed_text)
        parsed_text = re.sub('([0-9]+)', '', parsed_text)
        parsed_text = re.sub(underscore, ' ', parsed_text)

        return parsed_text


    def _split_into_tokens(self, text:str, flatten:bool):
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
        if flatten:
            self._tokens = [token for item in self._tokens for token in item]

        return self._tokens

    def _stopwords(self, doc:list):
        # Remove stopwords:
        return list(filter(lambda x: not x in self.stop_words, doc))

    def _normalize(self, doc:list):
        # Normalize the tokens with lemmatization
        normalized_tokens = list(map(lambda x: self.lematizer.lemmatize(x, pos='v'), doc))
        normalized_tokens = list(map(lambda x: self.lematizer.lemmatize(x, pos='n'), normalized_tokens))
        normalized_tokens = list(map(lambda x: self.lematizer.lemmatize(x, pos='a'), normalized_tokens))

        return normalized_tokens

    def get_tokens(self):
        """
        Returns the list of all tokens after applying stopword removal and normalization
        """
        if self._df is not None:
            if self.stopwords and self.normalized:
                for index, doc in enumerate(self.list_docs):
                    self._split_into_tokens(text=doc, flatten=False)

                    self._tokens[index] = self._stopwords(self._tokens[index])
                    self._tokens[index] = self._normalize(self._tokens[index])
                    self._tokens[index] = list(filter(lambda x: len(x) > 1, self._tokens[index]))

            else:
                [self._split_into_tokens(doc, False) for doc in self.list_docs]


        else:
            self._split_into_tokens(self.full_text, flatten=True)
            if self.stopwords:
                self._tokens = self._stopwords(self._tokens)
            if self.normalized:
                self._tokens = self._normalize(self._tokens)

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
