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


def extract_repr_random_token_bert_mlm(tokenizer, data, output_embeddings=None):
    """
    Extract a representation in context (or an embedding) for a random token in each sentence, and add it to the data.

    :param model: mBERT model
    :param tokenizer: mBERT tokenizer
    :param data: the data that includes all sentences, representations will be added here
    :param output_embeddings: out of mBERT
    :return: data, with the added representations
    """
    for i, d in tqdm(enumerate(data), total=len(data)):
        sentence = d["text"]

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


    return data



def create_repr_per_lang(vecs, labels_lang):
    # create an average representation for each language
    lang_specific = vecs

    vectors = {}
    for vec, lang in zip(lang_specific, labels_lang):
        if lang not in vectors:
            vectors[lang] = []
        vectors[lang].append(vec)

    lang_repr = {}
    for lang in vectors:
        lang_repr[lang] = np.average(vectors[lang], axis=0)

    return lang_repr


def extract_north_euralex(eur2ted):
    """
    Extract data from north_euralex dataset, for the languages with which we trained INLP
    :param eur2ted: mapping of names of languages from north_euralex to TED
    :return: translation to different languages fro each En word in the dataset
    """

    # extract all words per concept in the data
    words = {}
    with open("../data/northeuralex-0.9-forms.tsv") as f_words:
        reader = csv.reader(f_words, delimiter="\t")
        for row in reader:
            if row[0] in eur2ted:
                concept = row[2]
                word = row[3]
                lang = row[0]
                if concept not in words:
                    words[concept] = []
                words[concept].append((word, lang))

    # reorganize such that we have all translations per En word
    map_word_pos = {}
    all_translations = {}
    for concept in words:
        translations = {}
        pos = concept.rsplit("::")[1]
        for pair in words[concept]:
            w, lang = pair
            translations[lang] = w

        all_translations[concept] = translations
        map_word_pos[concept] = pos

    return all_translations, map_word_pos


def sort_vocab(embed, output_embeddings):
    best = (np.argsort(np.dot(output_embeddings, embed)))[::-1]
    return list(best)


def sort_vocab_batch(embed, output_embeddings):
    best = (np.argsort(np.dot(output_embeddings, embed.T), axis=0))
    return list(best.T)



def evaluate_northeuralex_all_langs(source_lang, target_lang, lang_repr, eur2ted, all_translations, output_embeddings, tokenizer):
    """
    Evaluate the method using north_euralex.

    :return: ranks of the correct translation before and after.
    """
    rank_before = []
    rank_after = []

    for concept in tqdm(all_translations):

        translations = all_translations[concept]
        if source_lang not in translations or target_lang not in translations:
            continue
        source_word = translations[source_lang]
        source_id = tokenizer.encode(source_word, add_special_tokens=False)
        if len(source_id) != 1:
            # skip source words that are longer than one token
            continue
        source_id = source_id[0]

        trans_id_list = []
        target_lang_list = []
        source_embed_list = []
        new_source_embed_list = []

        trans = translations[target_lang]
        trans_id = tokenizer.encode(trans, add_special_tokens=False)

        if len(trans_id) != 1:
            # skip target words that are longer than one token
            continue
        trans_id = trans_id[0]
        trans_id_list.append(trans_id)
        target_lang_list.append(target_lang)

        source_embed = output_embeddings[source_id]
        
        # create new representation
        new_source_embed = source_embed - lang_repr[eur2ted[source_lang]] + lang_repr[eur2ted[target_lang]]

        # save to compute in batches
        source_embed_list.append(source_embed)
        new_source_embed_list.append(new_source_embed)

        # compute predictions before and after analogies
        if source_embed_list != []:
            before = sort_vocab_batch(np.array(source_embed_list), output_embeddings)
            for trans_id, ranks in zip(trans_id_list, before):
                rank_trans = list(ranks)[::-1].index(trans_id)
                rank_source = list(ranks)[::-1].index(source_id)
                if rank_source < rank_trans:
                    rank_trans -= 1
                rank_before.append(rank_trans)

            after = sort_vocab_batch(np.array(new_source_embed_list), output_embeddings)
            for trans_id, ranks in zip(trans_id_list, after):
                rank_trans = list(ranks)[::-1].index(trans_id)
                rank_source = list(ranks)[::-1].index(source_id)
                if rank_source < rank_trans:
                    rank_trans -= 1
                rank_after.append(rank_trans)

    return rank_before, rank_after


def evaluate_northeuralex(lang_repr, eur2ted, all_translations, map_word_pos, output_embeddings, tokenizer,
                          eval_pos, repr_filename=None, details_filename=None):
    """
    Evaluate the method using north_euralex.

    :return: ranks of the correct translation before and after.
    """
    rank_before = {}
    rank_after = {}
    all_new_repr = []
    all_before_repr = []
    all_details = []
    for concept in tqdm(all_translations):

        en_word = all_translations[concept]["eng"]

        en_id = tokenizer.encode(en_word, add_special_tokens=False)
        if len(en_id) != 1:
            # skip source words that are longer than one token
            continue

        if eval_pos and eval_pos != map_word_pos[concept]:
            continue

        en_id = en_id[0]

        translations = all_translations[concept]

        trans_id_list = []
        lang_list = []
        en_embed_list = []
        new_en_embed_list = []

        for lang in translations:
            if lang == "eng":
                continue


            if lang not in rank_before:
                rank_before[lang] = []
                rank_after[lang] = []


            trans = translations[lang]
            trans_id = tokenizer.encode(trans, add_special_tokens=False)

            if len(trans_id) != 1:
                # skip target words that are longer than one token
                continue
            trans_id = trans_id[0]
            trans_id_list.append(trans_id)
            lang_list.append(lang)

            en_embed = output_embeddings[en_id]

            # create new representation
            new_en_embed = en_embed - lang_repr["en"] + lang_repr[eur2ted[lang]]

            # save to compute in batches
            en_embed_list.append(en_embed)
            new_en_embed_list.append(new_en_embed)

            all_before_repr.append(en_embed)
            all_new_repr.append(new_en_embed)
            all_details.append((concept, en_word, en_id, lang, trans, trans_id))



        # compute predictions before and after analogies
        if en_embed_list != []:
            before = sort_vocab_batch(np.array(en_embed_list), output_embeddings)
            for trans_id, lang, ranks in zip(trans_id_list, lang_list, before):
                rank_trans = list(ranks)[::-1].index(trans_id)
                rank_en = list(ranks)[::-1].index(en_id)
                if rank_en < rank_trans:
                    rank_trans -= 1
                rank_before[lang].append(rank_trans)

            after = sort_vocab_batch(np.array(new_en_embed_list), output_embeddings)
            for trans_id, lang, ranks in zip(trans_id_list, lang_list, after):
                rank_trans = list(ranks)[::-1].index(trans_id)
                rank_en = list(ranks)[::-1].index(en_id)
                if rank_en < rank_trans:
                    rank_trans -= 1
                rank_after[lang].append(rank_trans)
    assert (len(all_new_repr) == len(all_details))

    # save representations and details to files
    np.save(repr_filename, np.array(all_new_repr))
    np.save(repr_filename+"_before", np.array(all_before_repr))
    with open(details_filename, "wb") as f:
        pickle.dump(all_details, f)

    return rank_before, rank_after


def acc_at_k(ranks, k=10):
    acc = 0.0
    for r in ranks:
        if r < k:
            acc += 1
    return acc / len(ranks)


def print_evals(ranks_a, ranks_b, return_accs=False):
    """
    Prints the different metrics.
    :param ranks_a: ranks before using method
    :param ranks_b: ranks after using method
    """
    assert (len(ranks_a) == len(ranks_b))

    print("hard win:", sum([1 for i, j in zip(ranks_a, ranks_b) if i > j]), "out of ", len(ranks_a))
    print("soft win:", sum([1 for i, j in zip(ranks_a, ranks_b) if i >= j]), "out of ", len(ranks_a))

    if len(ranks_a) == 0:
        return

    print("mean before:", mean(ranks_a))
    print("mean after:", mean(ranks_b))

    log_a = [np.log(i) if i > 0 else 0 for i in ranks_a]
    log_b = [np.log(i) if i > 0 else 0 for i in ranks_b]

    print("mean log before:", mean(log_a))
    print("mean log after:", mean(log_b))

    for k in [1, 5, 10, 50, 100]:
        print("acc@" + str(k) + ":", format(acc_at_k(ranks_a, k), '.3f'), format(acc_at_k(ranks_b, k), '.3f'))


    if return_accs:
        return acc_at_k(ranks_b, 1), acc_at_k(ranks_b, 5), acc_at_k(ranks_b, 10)



def check_pos(nlp, sentence, idx, pos):
    doc_partial = nlp(" ".join(sentence.split()[:idx]))
    idx = len(doc_partial)
    doc = nlp(sentence)

    if doc[idx].pos_ == pos:
        return True
    return False


def extract_pos(all_aligned, nlp, pos, max_sents):
    aligned_pos = []
    for item in all_aligned:
        if len(aligned_pos) >= max_sents:
            break
        if check_pos(nlp, item["en_sent"], item["idx"], pos):
            aligned_pos.append(item)

    return aligned_pos


def data_for_lang_repr(data, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    random.shuffle(data)
    vecs = np.array([d["vec"] for d in data if d["vec"] is not None])
    labels_lang = np.array([d["lang"] for d in data if d["vec"] is not None])

    return vecs, labels_lang


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_data", action="store_true", help="extract data from scratch (otherwise, load data from saved file)")
    parser.add_argument("--all_langs", action="store_true", help="Translate from all langs to all langs")
    parser.add_argument("--eval_pos", help="evaluate a specific pos tag: NOUN, VERB or ADJ")
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
    data_filename = "../data/data_with_states_{}lang_{}perlang_embeddings".format(num_langs, sent_per_lang)

    if args.extract_data:
        # extract the representations and dump them to a file
        data_with_states = extract_repr_random_token_bert_mlm(tokenizer_mlm,
                                                              copy.deepcopy(data)[:num_langs * sent_per_lang],
                                                              output_embeddings)
        with open(data_filename, "wb") as f:
            pickle.dump(data_with_states, f)
        print("extracted data")
    else:
        # load the representations instead of extracting them
        with open(data_filename, "rb") as f:
            data_with_states = pickle.load(f)
        print("loaded data")

    # data for lang_repr
    vecs, labels_lang = data_for_lang_repr(data_with_states, random_seed)

    # create a vector representation for each language, and save to file
    random.seed(random_seed)
    lang_repr = create_repr_per_lang(vecs, labels_lang)
    lang_repr_filename = "../data/lang_repr_{}lang_no_inlp".format(num_langs)

    with open(lang_repr_filename, "wb") as f:
        pickle.dump(lang_repr, f)


    # evaluation using northeuralex

    # extract north_euralex data and save
    # 'zh-tw', 'zh-cn' (chinese) and 'pt-br' (brazilian portuguese) are not in this data
    ted2eur = {"en": "eng", "ar": "arb", "he": "heb", "ru": "rus", "ko": "kor", "it": "ita", "ja": "jpn",
               "es": "spa", "fr": "fra", "nl": "nld", "ro": "ron", "tr": "tur"}

    eur2ted = {v: k for k, v in ted2eur.items()}
    all_translations, map_word_pos = extract_north_euralex(eur2ted)

    with open("../data/all_translations_north_euralex", "wb") as f:
        pickle.dump(all_translations, f)

    # names of files
    details_filename = "../data/north_euralex_details"
    repr_filename = "../data/representations_embed_{}lang_no_inlp".format(num_langs)

    # evaluate on north_euralex
    if args.all_langs:
        all_langs_1 = []
        all_langs_5 = []
        all_langs_10 = []
        for source_lang in ["eng", "rus", "nld", "fra", "spa", "ita", "ron", "tur", "kor", "jpn", "arb", "heb"]:
            for target_lang in ["eng", "rus", "nld", "fra", "spa", "ita", "ron", "tur", "kor", "jpn", "arb", "heb"]:
                print(source_lang, "to", target_lang)
                rank_before, rank_after = evaluate_northeuralex_all_langs(source_lang, target_lang, lang_repr,
                                                                          eur2ted, all_translations,
                                                                          output_embeddings, tokenizer_mlm)
                acc1, acc5, acc10 = print_evals(rank_before, rank_after, return_accs=True)
                all_langs_1.append(acc1)
                all_langs_5.append(acc5)
                all_langs_10.append(acc10)
        all_langs_1 = np.array(all_langs_1)
        all_langs_5 = np.array(all_langs_5)
        all_langs_10 = np.array(all_langs_10)
        print(all_langs_1)
        print(all_langs_5)
        print(all_langs_10)
        np.save("../data/all_langs_1", all_langs_1.reshape(12, 12))
        np.save("../data/all_langs_5", all_langs_5.reshape(12, 12))
        np.save("../data/all_langs_10", all_langs_10.reshape(12, 12))

    else:
        rank_before, rank_after = evaluate_northeuralex(lang_repr, eur2ted, all_translations, map_word_pos, output_embeddings,
                                                            tokenizer_mlm,
                                                            args.eval_pos,
                                                            repr_filename=repr_filename,
                                                            details_filename=details_filename)


        # print evaluations
        for lang in rank_before:
            print("\nlang:", lang)
            print_evals(rank_before[lang], rank_after[lang])

        rank_before_all = []
        rank_after_all = []
        for lang in rank_before:
            rank_before_all += rank_before[lang]
            rank_after_all += rank_after[lang]
        print("\nall together\n")
        print_evals(rank_before_all, rank_after_all)



if __name__ == "__main__":
    main()
