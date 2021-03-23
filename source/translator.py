import random
import torch
from transformers import BertTokenizer, BertForMaskedLM
import csv
import numpy as np
from transformers import *
from tqdm import tqdm
import pickle


def extract_lang_mapping(short_names, tokenizer):
    # get mapping from ISO to language names (based on north_euralex)
    # remove languages whose names are longer than a single token
    mapping = {}
    with open("../data/northeuralex-0.9-language-data.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[0] != "name":
                language = row[0]
                if language in short_names:
                    language = short_names[language]
                if len(language.split()) > 1:
                    continue
                if len(tokenizer.tokenize(language)) > 1:
                    continue
                iso = row[2]
                mapping[iso] = language
    return mapping


def get_repr_from_template(sentence, tokenizer, model):
    # extract representation of masked token

    prefix, mask, suffix = sentence.split("***")
    prefix_tokens = tokenizer.tokenize(prefix)
    suffix_tokens = tokenizer.tokenize(suffix)
    tokens = ['[CLS]'] + prefix_tokens + ['[MASK]'] + suffix_tokens + ['[SEP]']
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    target_idx = len(prefix_tokens) + 1

    with torch.no_grad():
        hidden_states = model(input_ids)[1]  # hidden_states, len=13
        last_layer = hidden_states[-1][0]
        word_last_layer = last_layer[target_idx]
        cls_layer = model.cls.predictions.transform(word_last_layer)
    return cls_layer.numpy()


def main():

    random_seed = 10
    random.seed(random_seed)

    pretrained_weights = 'bert-base-multilingual-uncased'
    tokenizer_mlm = BertTokenizer.from_pretrained(pretrained_weights)
    model_mlm = BertForMaskedLM.from_pretrained(pretrained_weights, output_hidden_states=True)
    model_mlm.eval()

    short_names = {"Modern Hebrew": "Hebrew", "Modern Greek": "Greek", "Western Farsi": "Farsi",
                   "Standard Albanian": "Albanian", "Standard Arabic": "Arabic", "Norwegian (Bokmål)": "Norwegian"}
    mapping = extract_lang_mapping(short_names, tokenizer_mlm)

    with open("../data/all_translations_north_euralex", "rb") as f:
        all_translations = pickle.load(f)

    template = "The word '{}' in {} is: ***mask***."

    all_repr = []
    repr_filename = "../data/representations_embed_translator_template"
    for concept in tqdm(all_translations):

        en_word = all_translations[concept]["eng"]
        en_id = tokenizer_mlm.encode(en_word, add_special_tokens=False)
        if len(en_id) != 1:
            # skip source words that are longer than one token
            continue

        translations = all_translations[concept]

        for lang in translations:
            if lang == "eng":
                continue
            trans = translations[lang]
            trans_id = tokenizer_mlm.encode(trans, add_special_tokens=False)
            if len(trans_id) != 1:
                # skip target words that are longer than one token
                continue

            # extract representation of masked token
            sentence = template.format(en_word, mapping[lang])
            repr = get_repr_from_template(sentence, tokenizer_mlm, model_mlm)
            all_repr.append(repr)

    np.save(repr_filename, np.array(all_repr))


if __name__ == "__main__":
    main()

