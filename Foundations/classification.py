
import pandas as pd
import numpy as np
from Foundations.foundations import Tokenizer
from embeddings import VectorEmbeddings
from sklearn.linear_model import LogisticRegression

from torch.utils.data import Dataset, DataLoader
import torch
import os

DATA_ROOT_DIR = '../data'
FILE_NAME = 'hamlet.txt'
SERIES = 'data.csv'

class EmbeddingsDataSet(Dataset):
    def __init__(
            self,
            root_dir:str,
            file_name:str,
            attribute:str,
            label:str,
            train:bool,
            max_docs:int=100
        ):

        self.root_dir = root_dir,
        self.filename = file_name,
        self.max_docs = max_docs
        self.train_state = train

        if not os.path.exists(os.path.join(root_dir, file_name)):
            raise ValueError(
                f'The given path: {os.path.join(root_dir, file_name)} does not exist'
            )

        if self.filename[0].endswith('.csv'):
            if self.train_state:
                self.docs = pd.read_csv(os.path.join(root_dir, file_name))[attribute][:40000]
                self.labels = pd.read_csv(os.path.join(root_dir, file_name))[label][:40000]
                self.embedding_model = VectorEmbeddings(
                    series=self.docs,
                    embedding='bert',
                )
            else:
                self.docs = pd.read_csv(os.path.join(root_dir, file_name))[attribute][-10000:]
                self.labels = pd.read_csv(os.path.join(root_dir, file_name))[label][-10000:]
                self.embedding_model = VectorEmbeddings(
                    series=self.docs,
                    embedding='bert',
                )
                self.embedding_model.load_model()
        else:
            raise ValueError(
                f'The give file: {self.filename} is not a valid csv file'
            )

    def __len__(self):
        if self.docs is not None:
            return len(self.docs)

    def __getitem__(self, idx):
        if self.docs is not None:
            doc = self.docs[idx]
            text_label = self.labels[idx]
            if text_label == 'positive':
                label = torch.tensor(0)
            else:
                label = torch.tensor(1)
            embedding = self.embedding_model.infer_vector(doc)
            sample = (embedding, label)

            return sample


class LogisticRegressionClassifier(torch.nn.Module):
    def __init__(
            self,
            input_dim:int,
            output_dim:int
        ):
        super(LogisticRegressionClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


class Classifier():
    def __init__(
        self,
        train_docs:EmbeddingsDataSet,
        classifier:torch.nn.Module,
        loss,
        optim:torch.optim.Optimizer,
        epochs:int
    ):
        self.train_docs = train_docs
        self.classifier = classifier
        self.loss = loss
        self.optim = optim
        self.epochs = epochs


    def _prepare_data(self):
        self.embedding_dataloader = DataLoader(
            self.train_docs,
            batch_size=50,
            num_workers=1
        )

    def train_classifier(self):
        self._prepare_data()
        for epoch in range(self.epochs):
            for i, (embedding, label) in enumerate(self.embedding_dataloader):
                embeds = embedding.requires_grad()
                self.optim.zero_grad()
                output = self.classifier(embedding)

    def predict(self, doc_index:int):
        doc_vect = self.document_vectorizer.infer_vector(self.documents, doc_index)
        return self.model.predict(
            doc_vect.reshape(1, -1)
        )



if __name__ == '__main__':
    embedding_dataset_train = EmbeddingsDataSet(
        root_dir=DATA_ROOT_DIR,
        file_name=SERIES,
        attribute='review',
        label='sentiment',
        train=True
    )

    embedding_dataset_test = EmbeddingsDataSet(
        root_dir=DATA_ROOT_DIR,
        file_name=SERIES,
        attribute='review',
        label='sentiment',
        train=False
    )