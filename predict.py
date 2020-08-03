import os
import re
import numpy as np
import pandas as pd
import pickle
import json
import lightgbm
from sklearn.model_selection import train_test_split
from bpemb import BPEmb
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# text_path = r'C:/Users/Dkrd/Documents/GitHub/writers_block/test_text'
#
# def import_text(path):
#     res_data = []
#     print('Read data...')
#     if len(os.listdir(path)):
#         for files in os.listdir(path):
#             print('>>>', files)
#             with open(os.path.join(path, files), 'r', encoding='utf-8') as f:
#                 res_data.append(f.read())
#     return res_data
#
# test_data = import_text(text_path)

bpemb_ru = BPEmb(lang='ru', vs=100000)
with open("vectorizer_word.pickle", "rb") as wordvec:
    vectorizer_word = pickle.load(wordvec)
with open("vectorizer_word.pickle", "rb") as charvec:
    vectorizer_char = pickle.load(charvec)
model = lightgbm.Booster(model_file='lightgbm_model.txt')

def predict(text):
    features = preprocess(text)
    y_pred = model.predict(features)
    pred_average = sum(y_pred)/len(y_pred)
    return pred_average

def preprocess(text):
    test_data = []
    test_data.append(text)
    X_test = pd.DataFrame(split_text(test_data[0], 300))
    X_test_tabfeatures = PrepareText(X_test)
    X_test = cleanText(X_test)
    X_test_tabfeatures['relative_unique_word_count'] = X_test[0].apply(
        lambda text: (len(set(w for w in text.split())) / (0.000001 + len(text.split()))))
    tabular_features_test = normalize(np.array(X_test_tabfeatures))
    text_test = np.array(X_test[0])
    test_features_word = vectorizer_word.transform(text_test)
    test_features_char = vectorizer_char.transform(text_test)
    test_features = hstack([tabular_features_test, test_features_char, test_features_word]).tocsr()
    return test_features

def split_text(text, subtext_length=300):
    cursor = 0
    subtexts = []
    textsplit = text.split()
    for i in range(9999999999999):
        if len(textsplit) - cursor > subtext_length:
            subtexts.append(' '.join(textsplit[cursor:cursor+subtext_length]))
            cursor += subtext_length
        else:
            subtexts.append(' '.join(textsplit[cursor:]))
            break
    return subtexts

def PrepareText(data):
    tabfeatures = pd.DataFrame()
    #Text's total length
    tabfeatures['total_length_count'] = data[0].apply(len)
    #Exclamation mark count
    tabfeatures['exclamation_count'] = data[0].apply(lambda text: text.count('!'))
    #Question mark count
    tabfeatures['question_count'] = data[0].apply(lambda text: text.count('?'))
    #Punctuation count
    tabfeatures['punctuation_count'] = data[0].apply(lambda text: sum(text.count(w) for w in '.,;:'))
    #Amount of upper case letters
    tabfeatures['uppercase_amount'] = data[0].apply(lambda text: sum(1 for c in text if c.isupper()))
    #Amount of upper case letters compared to the text's length
    tabfeatures['FULLCAPS_COUNT'] = tabfeatures['uppercase_amount']/tabfeatures['total_length_count']
    #Amount of unique words compared to total word count
    return tabfeatures

def cleanPunc(text):
    no_punctuation = re.sub(r'[?|!|\'|"|#]',r'',text)
    no_punctuation = re.sub(r'[.|,|)|(|\|/|—]',r' ',text)
    no_punctuation = no_punctuation.strip()
    no_punctuation = no_punctuation.replace("\n"," ")
    return no_punctuation

def cleanHtml(text):
    cleanr = re.compile('<.*?>')
    no_html = re.sub(cleanr, ' ', str(text))
    return no_html

def keepAlpha(text):
    alpha_sent = ""
    for word in text.split():
        alpha_word = re.sub('[^а-я А-Я]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def cleanText(data):
    data[0] = data[0].str.lower()
    data[0] = data[0].apply(cleanHtml)
    data[0] = data[0].apply(cleanPunc)
    data[0] = data[0].apply(keepAlpha)
    return data

def normalize(array):
  norm = np.linalg.norm(array)
  array = array/norm
  return array

def tokenizer(text):
    text = bpemb_ru.encode(text)
    return text

# def main():
#     result = predict(model, text)
#     print(result)

# if __name__ == "__main__":
#     print("allahu akbar")
#     main()
