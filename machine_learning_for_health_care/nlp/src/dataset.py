import os
import re
import string

import nltk
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec, phrases
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download(["stopwords", "omw-1.4", "wordnet"])


class Word2Vec():
    def __init__(self, corpus_file, vector_size, window, min_count, sg):
        self.corpus_file = corpus_file
        self.vector_size = vector_size  # Dimensionality of the word vectors.

        # Maximum distance between the current and predicted word within a sentence.
        self.window = window

        # Ignores all words with total frequency lower than this.
        self.min_count = min_count

        self.sg = sg  # Training algorithm: 1 for skip-gram; otherwise CBOW.
        self.model = Word2Vec(vector_size=self.vector_size,
                              min_count=self.min_count, sg=self.sg)

        self.label_to_vec = {"BACKGROUND":  0,
                             "OBJECTIVE":   1,
                             "METHODS":     2,
                             "RESULTS":     3,
                             "CONCLUSIONS": 4}

    def setup(self):
        self.model.build_vocab(self.corpus_file)
        #self.model.train(self.corpus_file, total_examples=self.model.corpus_count, epochs=self.model.epochs)

        bigram_transformer = phrases.Phrases(
            self.corpus_file, min_count=self.min_count)
        #self.model.train(bigram_transformer[self.corpus_file], total_examples= self.model.corpus_count, epochs=self.model.epochs)

        trigram_transformer = phrases.Phrases(
            bigram_transformer[self.corpus_file])
        self.model.train(trigram_transformer[self.corpus_file],
                         total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def sentence_to_vector(self, sentence: list):
        vector = np.zeros(self.vector_size)
        for word in sentence:
            vector += self.model.wv[word]
        vector = vector / self.vector_size
        return vector

    def getVectors(self, df):
        df["Labels"] = [self.label_to_vec[label] for label in df["Labels"]]
        df["Sentences"] = [self.sentence_to_vector(
            sentence) for sentence in df["Sentences"]]

        return torch.tensor(df["Sentences"]), torch.tensor(df["Labels"])


class PreprocessData():
    def __init__(self, data_dir="data", dataset="PubMed_20k_RCT", lower=True, rem_stop_words=True, stemming=False, lemmatisation=False):
        self.data_dir = data_dir
        self.dataset = dataset
        self.lower = lower
        self.rem_stop_words = rem_stop_words
        self.stemming = stemming
        self.lemmatisation = lemmatisation

        self.out_dir = os.path.join(self.data_dir, "processed_" + self.dataset)
        os.makedirs(self.out_dir, exist_ok=True)

    def read(self, file_name="dev"):
        path = os.path.join(self.data_dir, self.dataset, file_name + ".txt")
        labels, sentences = [], []
        with open(path, "r") as f:
            for line in f.readlines():
                if not line.startswith("#") and line.strip() != "":
                    label, sentence = line.split("\t")
                    labels.append(label)
                    sentences.append(sentence)

        return labels, sentences

    def preprocess(self, sentences: list[str]):
        if self.lower:
            sentences = [s.lower() for s in sentences]

        # remove punctuations
        sentences = [s.translate(str.maketrans(
            "", "", string.punctuation)) for s in sentences]

        # split sentences to word tokens
        sentences = [s.split() for s in sentences]

        if self.rem_stop_words:
            stop_words = set(stopwords.words('english'))
            sentences = [[w for w in s if not w.lower() in stop_words]
                         for s in sentences]

        if self.stemming:
            ps = PorterStemmer()
            sentences = [[ps.stem(w) for w in s] for s in sentences]

        if(self.lemmatisation):
            lz = WordNetLemmatizer()
            sentences = [[lz.lemmatize(w) for w in s] for s in sentences]

        sentences = [" ".join(s) for s in sentences]
        return sentences

    def load(self, file):
        labels, sentences = self.read(file)
        sentences = self.preprocess(sentences)
        df = pd.DataFrame({"labels": labels, "sentences": sentences})
        df["sentences"].replace("", np.nan, inplace=True)
        df.dropna(subset=["sentences"], inplace=True)
        df.to_csv(os.path.join(self.out_dir, file + ".csv"), index=False)
        return df

    def createFiles(self):
        print("Start preprocessing")
        dev = self.load("dev")
        train = self.load("train")
        test = self.load("test")
        print("Finished preprocessing")
        return dev, train, test
