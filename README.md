# It’s not Greek to mBERT:  <br/>  Inducing Word-Level Translations from Multilingual BERT

This project includes the experiments described in the [paper](https://www.aclweb.org/anthology/2020.blackboxnlp-1.5.pdf): 

**"It’s not Greek to mBERT: Inducing Word-Level Translations from Multilingual BERT"** 

Hila Gonen, Shauli Ravfogel, Yanai Elazar, Yoav Goldberg, BlackboxNLP, 2020.

## Prerequisites

* Python 3
* Required INLP code is already in this repository

## Data

Please download relevant data files from this [folder](https://drive.google.com/drive/folders/1sx7jMFHvzKB5zcE7aA-f6kMQhgyECS8C?usp=sharing)


## Rowspace (language specific) and Nullspace (language neutral) matrices

You can either create these matrices using the script **inlp-oop/create_inlp_matrix_lang.py**.

Example:

```
python create_inlp_matrix_lang.py --extract_data --train_P
```


Alternatively, you can download them from this [folder](https://drive.google.com/drive/folders/1Rgh1Eu02CJsQI6nESSy69XORqMoTpv31?usp=sharing):
                                                            

* rowspace_embeddings: Rowspace (language specific) for embeddings
* nullspace_embeddings: Nullspace (language neutral) for embeddings
* rowspace_context: Rowspace (language specific) for representations in context
* nullspace_context: Nullspace (language neutral) for representations in context

### Details:

* Languages used (15 most common in TED):
zh-tw, zh-cn, pt-br, en, ar, he, ru, ko, it, ja, es, fr, nl, ro, tr

* 20 iterations of INLP

* 5000 examples per language

* Embeddings matrices: embedding representations of a random token from each sentence, excluding token that start with "##"

* In-context matrices: representation in context of a random token from each sentence, may include "CLS" or "SEP" tokens


## Analogies method

Use the script **source/analogies.py**.


## Template-based method

Use the script **source/translator.py**.
This will create a file with the matching representations for the Euralex data: **../data/representations_embed_translator_template.npy** that will be used in **source/compare_translator_analogies.py** for evaluation.

## Misc.

For comparison between the methods, use **source/compare_translator_analogies.py**.

For language prediction, use the notebook **source/translator_langs.ipynb**.

## Cite

If you find this project useful, please cite the paper:
```
@inproceedings{gonen_mbert20,
    title = "It{'}s not {G}reek to m{BERT}: Inducing Word-Level Translations from Multilingual {BERT}",
    author = "Gonen, Hila and Ravfogel, Shauli and Elazar, Yanai and Goldberg, Yoav",
    booktitle = "Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    year = "2020",
}
```

## Contact

If you have any questions or suggestions, please contact [Hila Gonen](mailto:hilagnn@gmail.com).

## License

This project is licensed under Apache License - see the [LICENSE](LICENSE) file for details.


