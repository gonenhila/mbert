import random
import torch
import tensorflow as tf
from transformers import BertTokenizer, BertForMaskedLM
import csv
import time
import numpy as np
from transformers import *
from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm
import copy
import inlp_dataset_handler
import inlp
import inlp_linear_model
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, SGDClassifier
from sklearn.svm import LinearSVC, SVR
import pandas as pd
from collections import Counter
from tqdm import tqdm
import pickle
import jsonlines
from statistics import mean
import argparse
import collections
import spacy


def collect_data_per_lang(num_langs=15, sent_per_lang=5000):
    """
    collect sentences for each language
    :param num_langs: number of langs to extract sentences for
    :param sent_per_lang: number of sentences to extract per lang
    :return: data: the sentences, langs: the most frequent langs
    """

    # collect sentences for each language
    df = pd.read_csv("../data/all_talks_train.tsv", sep="\t")
    df = df[df["en"].str.len() > 75]
    df = df.dropna()
    del df["talk_name"]
    all_langs = list(df.columns.values)

    all_data = []
    i = 0
    for lang in all_langs:
        relevant = df[lang].tolist()
        for sent in relevant:
            if "NULL" not in sent:
                all_data.append({"text": sent.replace("\n", ".").replace("&apos;", "'"), "lang": lang, "id": i})
                i += 1

    # find frequent languages
    lang_counter = Counter()
    for lang in all_langs:
        relevant = [d for d in all_data if d["lang"] == lang]
        lang_counter[lang] = len(relevant)

    common_langs = lang_counter.most_common(num_langs)
    common_langs, _ = list(zip(*common_langs))
    langs = common_langs

    # collect data (<sent_per_lang> sentences) only from <num_langs> most frequent languages
    data = []
    used_sentences = []
    n = sent_per_lang
    for lang in langs:
        lang_sentences = [d for d in all_data if d["lang"] == lang]
        for i in range(n):
            sentence = random.choice(lang_sentences)
            data.append({"lang": lang, "text": sentence["text"]})
            used_sentences.append(sentence["id"])

    with open("../data/used_sentences", "wb") as f:
        pickle.dump(used_sentences, f)

    random.shuffle(data)
    return data, langs


def extract_repr_random_token_bert_mlm(model, tokenizer, data, embeddings=False, output_embeddings=None):
    """
    Extract a representation in context (or an embedding) for a random token in each sentence, and add it to the data.

    :param model: mBERT model
    :param tokenizer: mBERT tokenizer
    :param data: the data that includes all sentences, representations will be added here
    :param embeddings: if True, extract embeddings instead of WiC representation
    :param output_embeddings: out of mBERT
    :return: data, with the added representations
    """
    for i, d in tqdm(enumerate(data), total=len(data)):
        sentence = d["text"]

        if embeddings:
            input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=False)])
            input_ids = input_ids.detach().cpu().numpy()
            input_ids = list(input_ids[0])
            random.shuffle(input_ids)
            for token in input_ids:
                if tokenizer.convert_ids_to_tokens([token])[0].startswith("##"):
                    continue
                embed = output_embeddings[token]
                assert (len(embed) == 768), len(embed)
                d["vec"] = embed
                break
            if "vec" not in d:
                d["vec"] = None

        else:
            input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
            idx = random.randrange(1, len(input_ids[0]))
            with torch.no_grad():
                hidden_states = model(input_ids)[1]  # hidden_states, len=13
                last_layer = hidden_states[-1][0]
                word_last_layer = last_layer[idx]
                cls_layer = model.cls.predictions.transform(word_last_layer)
                data[i]["vec"] = cls_layer.numpy()
                data[i]["idx"] = idx
                data[i]["tokens"] = input_ids

    return data


def extract_repr_specific_token_bert_mlm(sentence, idx, tokenizer, model):
    # extract the representation of a specific token from the last layer

    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])

    with torch.no_grad():
        hidden_states = model(input_ids)[1]  # hidden_states, len=13
        last_layer = hidden_states[-1][0]
        word_last_layer = last_layer[idx]
        cls_layer = model.cls.predictions.transform(word_last_layer)
    return cls_layer.numpy()


def data_for_classification(data, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    random.shuffle(data)
    vecs = np.array([d["vec"] for d in data if d["vec"] is not None])
    labels_lang = np.array([d["lang"] for d in data if d["vec"] is not None])
    l = int(len(vecs) * 0.9)
    x_train_cls, x_dev_cls = vecs[:l], vecs[l:]
    y_train_cls, y_dev_cls = labels_lang[:l], labels_lang[l:]
    model = SGDClassifier(max_iter=1000, alpha=0.01)
    model.fit(x_train_cls, y_train_cls)
    print(model.score(x_dev_cls, y_dev_cls))
    return vecs, labels_lang, x_train_cls, y_train_cls, x_dev_cls, y_dev_cls


def trigger_INLP(x_train_cls, y_train_cls, x_dev_cls, y_dev_cls, num_classifiers=20):
    inlp_dataset = inlp_dataset_handler.ClassificationDatasetHandler(x_train_cls, y_train_cls, x_dev_cls, y_dev_cls,
                                                                     dropout_rate=0, Y_train_main=None, Y_dev_main=None,
                                                                     by_class=False,
                                                                     equal_chance_for_main_task_labels=False)
    inlp_model_handler = inlp_linear_model.SKlearnClassifier(SGDClassifier, {"max_iter": 10000, "alpha": 0.0001,
                                                                             "n_iter_no_change": 10})
    P_nullspace, rowspace_projections, Ws = inlp.run_INLP(num_classifiers=num_classifiers, input_dim=768,
                                                          is_autoregressive=True,
                                                          min_accuracy=0, dataset_handler=inlp_dataset,
                                                          model=inlp_model_handler)
    P_rowspace = np.eye(P_nullspace.shape[0]) - P_nullspace

    return P_nullspace, P_rowspace




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_data", action="store_true", help="extract data from scratch (otherwise, load data from saved file)")
    parser.add_argument("--train_P", action="store_true", help="train P from scratch (otherwise, load P from saved file")
    args = parser.parse_args()

    random_seed = 10
    num_langs = 15
    sent_per_lang = 5000
    num_classifiers = 20
    num_to_eval = 1000
    random.seed(random_seed)

    print("random_seed", random_seed)
    print("num_langs", num_langs)
    print("sent_per_lang", sent_per_lang)
    print("num_classifiers", num_classifiers)
    print("num_to_eval", num_to_eval)
    print(args)

    # load mBERT
    pretrained_weights = 'bert-base-multilingual-uncased'
    tokenizer_mlm = BertTokenizer.from_pretrained(pretrained_weights)
    model_mlm = BertForMaskedLM.from_pretrained(pretrained_weights, output_hidden_states=True)
    output_embeddings = model_mlm.cls.predictions.decoder.weight.detach().cpu().numpy()

    # collect data (<sent_per_lang> sentences) from <num_langs> most frequent languages, from TED
    data, langs = collect_data_per_lang(num_langs=num_langs, sent_per_lang=sent_per_lang)

    # extract representations of random tokens from each sentence
    random.seed(random_seed)
    data_filename = "../data/data_with_states_{}lang_{}perlang".format(num_langs, sent_per_lang)
    if args.embeddings:
        data_filename = data_filename + "_embeddings"
    if args.extract_data:
        # extract the representations and dump them to a file
        data_with_states = extract_repr_random_token_bert_mlm(model_mlm, tokenizer_mlm,
                                                              copy.deepcopy(data)[:num_langs * sent_per_lang],
                                                              args.embeddings, output_embeddings)
        with open(data_filename, "wb") as f:
            pickle.dump(data_with_states, f)
        print("extracted data")
    else:
        # load the representations instead of extracting them
        with open(data_filename, "rb") as f:
            data_with_states = pickle.load(f)
        print("loaded data")

    # create data for INLP (and classification in general)
    vecs, labels_lang, x_train_cls, y_train_cls, x_dev_cls, y_dev_cls = data_for_classification(data_with_states,
                                                                                                random_seed)

    # get matrices from INLP
    random.seed(random_seed)
    np.random.seed(random_seed)

    null_matrix_name = "P_nullspace_{}lang_{}iter_{}perlang".format(num_langs, num_classifiers, sent_per_lang)
    row_matrix_name = "P_rowspace_{}lang_{}iter_{}perlang".format(num_langs, num_classifiers, sent_per_lang)
    if args.embeddings:
        null_matrix_name = null_matrix_name + "_embeddings"
        row_matrix_name = row_matrix_name + "_embeddings"

    # train INLP from scratch, and save matrices
    P_nullspace, P_rowspace = trigger_INLP(x_train_cls, y_train_cls, x_dev_cls, y_dev_cls,
                                           num_classifiers=num_classifiers)
    np.save("../data/" + null_matrix_name, P_nullspace)
    np.save("../data/" + row_matrix_name, P_rowspace)
    print("trained P")



if __name__ == "__main__":
    main()

