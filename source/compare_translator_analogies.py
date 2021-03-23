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
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
import pdb
import pdb

def get_predictions(repr, output_embeddings):
    scores = np.dot(output_embeddings, np.array(repr).T).T
    best = (np.argsort(scores, axis=1))[:, ::-1]
    return list(best)


def plot_tsne(vectors_a, vectors_b, labels=None, lang_repr=None, annotate=False, lang2label=None, eur2ted=None):
    vectors = []
    if not labels:
        labels = len(vectors_a) * [0]
    else:
        labels = labels
    for vec in vectors_a:
        vectors.append(vec)

    colors = cm.rainbow(np.linspace(0, 1, 12))

    last_label = len(set(labels))
    if lang_repr is not None:
        for lang in lang_repr:
            vectors.append(lang_repr[lang])
            labels.append(last_label)

    # perform TSNE
    X_embedded = TSNE(n_components=2, random_state=1).fit_transform(vectors)

    if annotate:
        for lang, label in lang2label.items():
            relevant = X_embedded[np.array(labels) == label, :]
            mean_x, mean_y = np.median(relevant, axis=0)
            plt.annotate(eur2ted[lang], (mean_x, mean_y), fontsize=14, weight='bold')

    for x, l in zip(X_embedded, labels):
        plt.scatter(x[0], x[1], marker='.', c=colors[l])

    if lang_repr is not None:
        lang_repr_points = [X_embedded[-len(lang_repr):]]
        txt = [lang for lang in lang_repr]
        for t, point in zip(txt, lang_repr_points[0]):
            plt.annotate(t, (point[0], point[1]))

    plt.savefig("../figs/tsne_translator.png")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--tsne", action="store_true", help="plot Tsne")
    parser.add_argument("--per_lang", action="store_true", help="print acc per lang")
    parser.add_argument("--lang_repr_filename", default="../data/lang_repr_15lang_no_inlp", help="analyze predictions")
    parser.add_argument("--before_filename", default="../data/representations_embed_15lang_no_inlp_before.npy", help="representations before analogies")
    parser.add_argument("--analogies_filename", default="../data/representations_embed_15lang_no_inlp.npy",
                        help="representations of analogies")
    parser.add_argument("--translator_filename", default="../data/representations_embed_translator_template.npy", help="representations of translator")
    args = parser.parse_args()
    print(args)

    before_filename = args.before_filename
    translator_filename = args.translator_filename
    analogies_filename = args.analogies_filename
    before = np.load(before_filename)
    translator = np.load(translator_filename)
    analogies = np.load(analogies_filename)

    # load details of north_euralex dataset
    with open("../data/north_euralex_details", "rb") as f:
        all_details = pickle.load(f)

    pretrained_weights = 'bert-base-multilingual-uncased'
    tokenizer_mlm = BertTokenizer.from_pretrained(pretrained_weights)
    model_mlm = BertForMaskedLM.from_pretrained(pretrained_weights, output_hidden_states=True)
    model_mlm.eval()

    with open(args.lang_repr_filename, "rb") as f:
        lang_repr = pickle.load(f)
    ted2eur = {"en": "eng", "ar": "arb", "he": "heb", "ru": "rus", "ko": "kor", "it": "ita", "ja": "jpn",
               "es": "spa", "fr": "fra", "nl": "nld", "ro": "ron", "tr": "tur"}
    eur2ted = {v: k for k, v in ted2eur.items()}

    if args.tsne:

        num_to_plot = 5000

        langs = [item[3] for item in all_details]
        label2lang = list(set(langs))
        lang2label = {lang: label for label, lang in enumerate(label2lang)}

        labels = [lang2label[lang] for lang in langs]

        plot_tsne(translator[:num_to_plot, :], analogies[:num_to_plot, :], labels, annotate=True, lang2label=lang2label, eur2ted=eur2ted)

        return

    random_seed = 10
    random.seed(random_seed)

    total_predictions = 1000

    output_embeddings = model_mlm.cls.predictions.decoder.weight.detach().cpu().numpy()

    repr_idx = 0

    if args.per_lang:
        ranks_before = {}
        ranks_translator = {}
        ranks_analogies = {}
        ranks_both = {}
    else:
        ranks_before = []
        ranks_translator = []
        ranks_analogies = []
        ranks_both = []

    acc_before = {}
    acc_translator = {}
    acc_analogies = {}
    acc_both = {}

    reprs_before = []
    reprs_translator = []
    reprs_analogies = []
    reprs_both = []
    concepts = []
    en_words = []
    en_ids = []
    langs = []
    trans_ids = []


    for item in tqdm(all_details):
        concept, en_word, en_id, lang, trans, trans_id = item
        concepts.append(concept)
        en_words.append(en_word)
        en_ids.append(en_id)
        langs.append(lang)
        trans_ids.append(trans_id)
        reprs_before.append(before[repr_idx])
        reprs_translator.append(translator[repr_idx])
        reprs_analogies.append(analogies[repr_idx])

        reprs_both.append(translator[repr_idx] - lang_repr["en"] + lang_repr[eur2ted[lang]])

        repr_idx += 1

        if len(trans_ids) == 16 or repr_idx == len(translator):

            # get predictions for current batch
            preds_before = get_predictions(reprs_before, output_embeddings, k=total_predictions)
            preds_translator = get_predictions(reprs_translator, output_embeddings, k=total_predictions)
            preds_analogies = get_predictions(reprs_analogies, output_embeddings, k=total_predictions)
            preds_both = get_predictions(reprs_both, output_embeddings, k=total_predictions)

            for concept, trans_id, before_list, translator_list, analogies_list, both_list, en_word, en_id, lang in zip(concepts, trans_ids,
                                                                                            preds_before, preds_translator,
                                                                                           preds_analogies, preds_both,
                                                                                           en_words, en_ids, langs):


                # find rank of translation for each method
                rank_en = list(before_list).index(en_id)
                rank_before = list(before_list).index(trans_id)
                if rank_en < rank_before:
                    rank_before -= 1

                rank_en = list(translator_list).index(en_id)
                rank_translator = list(translator_list).index(trans_id)
                if rank_en < rank_translator:
                    rank_translator -= 1

                rank_en = list(analogies_list).index(en_id)
                rank_analogies = list(analogies_list).index(trans_id)
                if rank_en < rank_analogies:
                    rank_analogies -= 1

                rank_en = list(both_list).index(en_id)
                rank_both = list(both_list).index(trans_id)
                if rank_en < rank_both:
                    rank_both -= 1

                if args.per_lang:
                    if lang not in ranks_before:
                        ranks_before[lang] = []
                        ranks_translator[lang] = []
                        ranks_analogies[lang] = []
                        ranks_both[lang] = []
                    ranks_before[lang].append(rank_before)
                    ranks_translator[lang].append(rank_translator)
                    ranks_analogies[lang].append(rank_analogies)
                    ranks_both[lang].append(rank_both)

                else:
                    ranks_before.append(rank_before)
                    ranks_translator.append(rank_translator)
                    ranks_analogies.append(rank_analogies)
                    ranks_both.append(rank_both)

            reprs_before = []
            reprs_translator = []
            reprs_analogies = []
            reprs_both = []
            concepts = []
            en_words = []
            en_ids = []
            langs = []
            trans_ids = []


    # print evaluations

    if args.per_lang:
        for lang in ranks_before:
            acc_before[lang] = {}
            acc_translator[lang] = {}
            acc_analogies[lang] = {}
            acc_both[lang] = {}
            for i in [1, 5, 10, 50, 100]:
                acc_before[lang][i] = sum([1 for item in ranks_before[lang] if item < i]) / len(ranks_before[lang])
                acc_translator[lang][i] = sum([1 for item in ranks_translator[lang] if item < i]) / len(ranks_translator[lang])
                acc_analogies[lang][i] = sum([1 for item in ranks_analogies[lang] if item < i]) / len(ranks_analogies[lang])
                acc_both[lang][i] = sum([1 for item in ranks_both[lang] if item < i]) / len(ranks_both[lang])
            print(lang)
            print("before")
            for k, v in acc_before[lang].items():
                print(k, format(v, '.3f'))
            print("translator")
            for k, v in acc_translator[lang].items():
                print(k, format(v, '.3f'))
            print("analogies")
            for k, v in acc_analogies[lang].items():
                print(k, format(v, '.3f'))
            print("both")
            for k, v in acc_both[lang].items():
                print(k, format(v, '.3f'))
            print("diff @10:", acc_translator[lang][10] - acc_analogies[lang][10])

    else:
        for i in [1, 5, 10, 50, 100]:
            acc_before[i] = sum([1 for item in ranks_before if item < i]) / len(ranks_before)
            acc_translator[i] = sum([1 for item in ranks_translator if item < i]) / len(ranks_translator)
            acc_analogies[i] = sum([1 for item in ranks_analogies if item < i]) / len(ranks_analogies)
            acc_both[i] = sum([1 for item in ranks_both if item < i]) / len(ranks_both)

        print("before\n")
        for k, v in acc_before.items():
            print(k, format(v, '.3f'))
        print("translator\n")
        for k, v in acc_translator.items():
            print(k, format(v, '.3f'))
        print("analogies\n")
        for k, v in acc_analogies.items():
            print(k, format(v, '.3f'))
        print("both\n")
        for k, v in acc_both.items():
            print(k, format(v, '.3f'))

        print("mean before:", format(mean(ranks_before), '.1f'))
        print("mean translator:", format(mean(ranks_translator), '.1f'))
        print("mean analogies:", format(mean(ranks_analogies), '.1f'))
        print("mean both:", format(mean(ranks_both), '.1f'))

        log_before = [np.log(i) if i > 0 else 0 for i in ranks_before]
        log_trans = [np.log(i) if i > 0 else 0 for i in ranks_translator]
        log_anag = [np.log(i) if i > 0 else 0 for i in ranks_analogies]
        log_both = [np.log(i) if i > 0 else 0 for i in ranks_both]

        print("mean log before:", format(mean(log_before), '.2f'))
        print("mean log translator:", format(mean(log_trans), '.2f'))
        print("mean log analogies:", format(mean(log_anag), '.2f'))
        print("mean log both:", format(mean(log_both), '.2f'))

        print("hard win trans:", format(sum([1 for i, j in zip(ranks_before, ranks_translator) if i > j])/float(len(ranks_translator)), '.3f'))
        print("soft win trans:", format(sum([1 for i, j in zip(ranks_before, ranks_translator) if i >= j])/float(len(ranks_translator)), '.3f'))
        print("hard win analog:", format(sum([1 for i, j in zip(ranks_before, ranks_analogies) if i > j])/float(len(ranks_analogies)), '.3f'))
        print("soft win analog:", format(sum([1 for i, j in zip(ranks_before, ranks_analogies) if i >= j])/float(len(ranks_analogies)), '.3f'))
        print("hard win both:", format(sum([1 for i, j in zip(ranks_before, ranks_both) if i > j])/float(len(ranks_both)), '.3f'))
        print("soft win both:", format(sum([1 for i, j in zip(ranks_before, ranks_both) if i >= j])/float(len(ranks_both)), '.3f'))



if __name__ == "__main__":
    main()
