# AraT5: Text-to-Text Transformers for Arabic Language Understanding and Generation

<img src="transfomer.png" alt="drawing" width="45%" height="45%" align="right"/>

# What is the repository is about?

This is the repository accompanying our paper [AraT5: Text-to-Text Transformers for Arabic Language Understanding and Generation](link). In this is the repository we introduce:
* Introduce **AraT5<sub>MSA</sub>**, **AraT5<sub>Tweet/sub>**, and **AraT5**: Three powerful Arabic-specific text-to-text Transformer based models.
* Introduce **ARGNE**:  A new benchmark for Arabic language generation and evaluation
* evaluate  ```AraT5``` models on ```ARGNE``` and compare against available language models.

Our models establish new state-of-the-art (SOTA) on  xx out of the yy datasets.
Our language models are publicaly available for research (see below).

The rest of this repository provides more information about our new language models, benchmark, and experiments.

---

## Table of Contents
- [1 Our Language Models](#1-Our-Language-Models)
  - [1.1 AraT5<sub>MSA</sub>](#11-AraT5--msa)
  - [1.2 AraT5<sub>Tweet</sub>](#12-AraT5--Tweet)
  - [1.3 AraT5](#13-AraT5)
  - [1.4 Training Data and Vocabulary](#14-training-data-and-vocabulary)
- [2. Our Benchmark: ARGNE](#2-our-benchmark-ARGNE)
  - [2.1 Machine Translation](#21-machine--translation)
  - [2.2 Text Summarization](#22-text-summarization)
  - [2.3 News Title Generation](#23-news-generation)
  - [2.4 Question Generation](#24-question-generation)


## 7. Citation
If you use our AraT5 models for your scientific publication, or if you find the resources in this repository useful, please cite our paper as follows (to be updated):
```
@inproceedings{abdul-mageed-etal-2021-arbert,
    title = "{AraT5: Text-to-Text Transformers for Arabic Language Understanding and Generation",
    author = "Nagoudi, El Moatez Billah  and
      Elmadany, AbdelRahim  and
      Abdul-Mageed, Muhammad",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.551",
    doi = "10.18653/v1/2021.acl-long.551",
    pages = "7088--7105",}

```

---

## 8. Acknowledgments
We gratefully acknowledge support from the Natural Sciences and Engineering Research Council  of Canada, the  Social  Sciences and  Humanities  Research  Council  of  Canada, Canadian  Foundation  for  Innovation,  [ComputeCanada](www.computecanada.ca) and [UBC ARC-Sockeye](https://doi.org/10.14288/SOCKEYE). We  also  thank  the  [Google TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) program for providing us with free TPU access.
