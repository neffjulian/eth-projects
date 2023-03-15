import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import yaml
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

from src.dataset import PreprocessData, Word2VecVectorizer

def print_metrics(pred_test, y_test, pred_train, y_train):
    print("test accuracy", str(np.mean(pred_test == y_test)))
    print("train accuracy", str(np.mean(pred_train == y_train)))
    print("\n Metrics and Confusion for SVM \n")
    print(metrics.confusion_matrix(y_test, pred_test))
    print(metrics.classification_report(y_test, pred_test))


def load_data(base_dir):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError("Dataset not found. Check if path is correct and data is already preprocessed")
    df_train = pd.read_csv(os.path.join(base_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(base_dir, "dev.csv"))
    df_test = pd.read_csv(os.path.join(base_dir, "test.csv"))
    return df_train, df_val, df_test

def main(cfg):

    if cfg.get("preprocess"):
        preprocessor = PreprocessData(**cfg)
        preprocessor.createFiles()

    data_dir = cfg.get("data_dir", "data")
    base_dir = os.path.join(data_dir, "processed_" + cfg["dataset"])
    df_train, df_val, df_test = load_data(base_dir)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train["Labels"])
    y_val = label_encoder.transform(df_val["Labels"])
    y_test = label_encoder.transform(df_test["Labels"])

    x_train, x_val, x_test = df_train["Sentences"], df_val["Sentences"], df_test["Sentences"]

    model = Pipeline([
        ("vect", TfidfVectorizer()),
        ("chi", SelectKBest(chi2, k=2000)),
        ("clf", RandomForestClassifier())
    ])

    # model = Pipeline([
    #     ("vect", Word2VecVectorizer(vector_size=1000, window=10, min_count=1, sg=1)),
    #     ("clf", RandomForestClassifier())
    # ])

    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    print_metrics(pred_test, y_test, pred_train, y_train)


def read_config_file(config_file):
    with open(config_file) as f:
        return yaml.load(f, yaml.FullLoader)


if __name__ == "__main__":
    model_parser = ArgumentParser(add_help=False)
    model_parser.add_argument("--config", help="path to config file")
    model_parser.add_argument("--model", help="model used in experiment", default="baseline")
    
    temp_args, _ = model_parser.parse_known_args()
    model_name = temp_args.model
    config = {}
    if temp_args.config is not None:
        config = read_config_file(temp_args.config)
        model_name = config.get("model")

    parser = ArgumentParser(parents=[model_parser])
    # if model_name:
    #     parser = MODEL_DICT[model_name].add_model_specific_args(parser)

    # add general expereminet parameters & training parameters
    parser.add_argument("--preprocess", help="Preprocess the data", action="store_true")

    parser = PreprocessData.add_preprocessor_args(parser)

    cli_args = parser.parse_args()
    args = config | vars(cli_args)
    main(args)
