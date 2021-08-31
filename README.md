# AraT5: Text-to-Text Transformers for Arabic Language Understanding and Generation

<img src="transfomer.png" alt="drawing" width="45%" height="45%" align="right"/>

# What is the repository is about?

This is the repository accompanying our paper [AraT5: Text-to-Text Transformers for Arabic Language Understanding and Generation](link). In this is the repository we introduce:
* Introduce **AraT5<sub>MSA</sub>**, **AraT5<sub>Tweet</sub>**, and **AraT5**: Three powerful Arabic-specific text-to-text Transformer based models;
* Introduce **ARGNE**:  A new benchmark for Arabic language generation and evaluation;
* evaluate  ```AraT5``` models on ```ARGNE``` and compare against available language models.

Our models establish new state-of-the-art (SOTA) on  xx out of the yy datasets.
Our language models are publicaly available for research (see below).

The rest of this repository provides more information about our new language models, benchmark, and experiments.

---

## Table of Contents
- [1 Our Language Models](#1-Our-Language-Models)
  - [1.1 Training Data](#11-training-data)
  - [1.2 Models Architecture](#12-models-architecture)
  - [1.3 AraT5 Models](#13-arat5-models)
- [2. Our Benchmark: ARGNE](#2-our-benchmark-ARGNE)
  - [2.1 Machine Translation](#21-machine-translation)
  - [2.2 Text Summarization](#22-text-summarization)
  - [2.3 News Title Generation](#23-news-generation)
  - [2.4 Question Generation](#24-question-generation)


## 1. Our Language Models



## 1.1 Training Data

* **MSA Training Data**: We use 70GB of MSA text 7.1B tokens) from the following sources: [AraNews](nagoudi2020machine), [El-Khair](elkhair-2016), [Gigaword](https://catalog.ldc.upenn.edu/LDC2009T30), [OSCAR](suarez2019asynchronous), [OSIAN](zeroual2019osian),  Wikipedia Arabic, and [Hindawi Books](https://www.hindawi.org/books/}{https://www.hindawi.org/books).

* **Twitter Training Data**: We randomly sample 1.5B Arabic tweets from a large in-house dataset of about 10B tweets. We use string matching to only include tweets with at least 3 Arabic words, regardless whether the tweet has non-Arabic string or not.  %That is, we do not remove non-Arabic so long as the tweet meets the $3$ Arabic word criterion. 
The dataset makes up 178GB of text 21B tokens. 


## 1.2 Models Architecture

To train our AraT5, we use the same architecture as ```T5-base``` [(Raffel 2019)](https://arxiv.org/abs/1910.10683) where both  encoder and decoder  has 12 layers
each with 12 attention heads, and 768 hidden units.


## 1.3 AraT5 Models


We pre-train three powerful variants of the text-to-text transformer (T5) model dedicated to Modern Standard Arabic (MSA) and Arabic dialects, AraT5. AraT5 comes. AraT5 comes in three flavors:  
*  **AraT5<sub>MSA</sub>**:  trained on MSA data exclusively
*  **AraT5<sub>Tweet</sub>**: trained on Twitter data (mix of MSA and dialectal Arabic), 
*  **AraT5**:  trained on both Twitter and MSA data.


## 2. Our Benchmark: ARGNE
To  evaluate  our  models, we  also introduce  **ARGNE**,   a new benchmark for   A new benchmark for Arabic language generation and evaluation.   ARGNE is composed of four tasks, namely, ```machine  translation```,  ```summarization```,  ```newstitle   generation```   and   ```question   generation```. ARGNE  is  collected  from  a  total  of  ten datasets, including two new large datasets pro-posed in this work.



### 2.1 Machine Translation

|**Reference**| **Data  (#classes)**     | **TRAIN**   | **DEV**    | **TEST**   |
|---------|--------|--------|-------|------|
|[Alomari et al. (2017)](https://www.researchgate.net/publication/317501447_Arabic_Tweets_Sentimental_Analysis_Using_Machine_Learning)|AJGT (2)      |   1.4K | -      |    361 | 
|[Abdul-Mageed et al. (2020b)](https://www.aclweb.org/anthology/2020.osact-1.3) |AraNET<sub>Sent</sub> (2)      | 100K | 14.3K | 11.8K |
|[Al-Twairesh et al. (2017)](https://www.aclweb.org/anthology/P16-1066)|AraSenTi (3)          |  11,117 |  1,407 |  1,382 | 
|[Abu Farha and Magdy (2017)](https://www.aclweb.org/anthology/2020.osact-1.5)|ArSarcasm<sub>Sent</sub> (3)   |   8.4K | -      |  2.K | 
|[Elmadany et al. (2018)](https://www.semanticscholar.org/paper/ArSAS-%3A-An-Arabic-Speech-Act-and-Sentiment-Corpus-Elmadany-Mubarak/d32d3bb226f1738f72c415c6b03b5ad66ff604a4)|ArSAS (3)                           |  24.7K | -      |  3.6K | 

## 7. Citation
If you use our AraT5 models for your scientific publication, or if you find the resources in this repository useful, please cite our paper as follows (to be updated):
```
@inproceedings{abdul-mageed-etal-2021-arbert,
    title = "{AraT5: Text-to-Text Transformers for Arabic Language Understanding and Generation",
    author = "Nagoudi, El Moatez Billah  and
      Elmadany, AbdelRahim  and
      Abdul-Mageed, Muhammad",
    booktitle = "https://arxiv.org/pdf/2104.07483",
    month = aug,
    year = "2021"}

```

---

## 8. Acknowledgments
We gratefully acknowledge support from the Natural Sciences and Engineering Research Council  of Canada, the  Social  Sciences and  Humanities  Research  Council  of  Canada, Canadian  Foundation  for  Innovation,  [ComputeCanada](www.computecanada.ca) and [UBC ARC-Sockeye](https://doi.org/10.14288/SOCKEYE). We  also  thank  the  [Google TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) program for providing us with free TPU access.
