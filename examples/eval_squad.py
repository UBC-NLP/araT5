## SQuAD evaluation script. Modifed slightly for this notebook

from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys

def arabic_clean_str(text):
    '''
    this method normalizes up an arabic string, currently not used in evaluation, but should be used in the future
    '''
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim
    text = text.strip()
    return text

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        # return re.sub(r'\b(a|an|the)\b', ' ', text)
        return re.sub('\sال^|ال', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return arabic_clean_str(white_space_fix(remove_articles(remove_punc(lower(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_squad(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
      total += 1
      exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
      f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


# from __future__ import print_function
# from collections import Counter
# import string
# import re
# import argparse
# import json
# import sys
# import nltk
# import random
# # nltk.download('punkt')
# from random import randint



# def normalize_answer(s):
#     """Lower text and remove punctuation, articles and extra whitespace."""

#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the)\b', ' ', text)
    
#     def remove_articles_ar(text):
#         return re.sub('\sال^|ال', ' ', text)

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles_ar(remove_articles(remove_punc(lower(s)))))


# def f1_score(prediction, ground_truth):
#     prediction_tokens = normalize_answer(prediction).split()
#     ground_truth_tokens = normalize_answer(ground_truth).split()
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1


# def exact_match_score(prediction, ground_truth):
#     return (normalize_answer(prediction) == normalize_answer(ground_truth))


# def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
#     scores_for_ground_truths = []
#     for ground_truth in ground_truths:
#         score = metric_fn(prediction, ground_truth)
#         scores_for_ground_truths.append(score)
#     return max(scores_for_ground_truths)


# def evaluate_squad(dataset, predictions):
#     f1 = exact_match = total = exact_sentence = inclusion = random = 0
#     for article in dataset:
#         for paragraph in article['paragraphs']:
#             for qa in paragraph['qas']:
#                 total += 1
#                 if qa['id'] not in predictions:
#                     message = 'Unanswered question ' + qa['id'] + \
#                               ' will receive score 0.'
#                     print(message, file=sys.stderr)
#                     continue
#                 ground_truths = list(map(lambda x: x['text'], qa['answers']))
#                 prediction = predictions[qa['id']]
#                 sents = nltk.sent_tokenize(paragraph['context'])
#                 indx_g = -1
#                 indx_p = -1
#                 i = 0
#                 for sent in sents:
#                     if sent.find(ground_truths[0]) != -1:
#                         indx_g = i
#                     if sent.find(prediction) != -1:
#                         indx_p = i
#                     i += 1
#                 test = randint(0,i)
#                 if test == indx_g:
#                     random += 1
#                 if prediction.find(ground_truths[0]) != -1 or ground_truths[0].find(prediction):
#                     inclusion += 1
#                 if indx_g == indx_p and indx_p != -1:
#                     exact_sentence += 1
#                 exact_match += metric_max_over_ground_truths(
#                     exact_match_score, prediction, ground_truths)
#                 f1 += metric_max_over_ground_truths(
#                     f1_score, prediction, ground_truths)
#     inclusion = inclusion / total
#     random = random / total
#     exact_sentence = 100 * exact_sentence / total
#     exact_match = 100.0 * exact_match / total
#     f1 = 100.0 * f1 / total

#     return {'exact_match': exact_match, 'f1': f1, 'exact_sentence': exact_sentence}

# predict_file=sys.argv[1]
# output_dir=sys.argv[2]
# with open(predict_file) as dataset_file:
#     dataset_json = json.load(dataset_file)
#     dataset = dataset_json['data']

# import glob, regex
# print ("steps_num"+"\t"+"exact_match"+"\t"+"f1"+"\t"+"exact_sentence")
# for ckpt in glob.glob(output_dir+"/predictions_epoch_*.json"):
#     # print (ckpt)
#     with open(ckpt) as prediction_file:
#         predictions = json.load(prediction_file)
#         eval_res = evaluate(dataset, predictions)
#         # print (eval_res)
#         steps_num=regex.sub("predictions_", "", ckpt.split("/")[-1].split(".")[0])
#         print (str(steps_num)+"\t"+ str(eval_res["exact_match"])+"\t"+str(eval_res["f1"])+"\t"+ str(eval_res["exact_sentence"]))