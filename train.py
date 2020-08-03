import os
import re
import numpy as np
import pandas as pd
import pickle
import json
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from bpemb import BPEmb
from scipy.sparse import hstack

train_data_path = r'C:/Users/Dkrd/Documents/GitHub/writers_block/data/train'
test_data_path = r'C:/Users/Dkrd/Documents/GitHub/writers_block/data/test'
bpemb_ru = BPEmb(lang='ru', vs=100000)

def train_model(train_data_path, test_data_path):
    train_data_path = train_data_path
    test_data_path = test_data_path

    train_data = pd.DataFrame(read_data(train_data_path))
    test_data = pd.DataFrame(read_data(test_data_path))

    train_hem_texts, train_hem_labels = split_text(train_data[0][0], 300, 1)
    train_other_texts, train_other_labels = split_text(train_data[0][1], 300, 0)

    test_hem_texts, test_hem_labels = split_text(test_data[0][0], 300, 1)
    test_other_texts, test_other_labels = split_text(test_data[0][1], 300, 0)

    X_train_all = pd.DataFrame(np.append(np.array(train_hem_texts), np.array(train_other_texts)))
    y_train_all = np.append(np.array(train_hem_labels), np.array(train_other_labels))

    X_test = pd.DataFrame(np.append(np.array(test_hem_texts), np.array(test_other_texts)))
    y_test = np.append(np.array(test_hem_labels), np.array(test_other_labels))

    X_train_tabfeatures = get_tab_features(X_train_all)
    X_test_tabfeatures = get_tab_features(X_test)

    X_train_all = cleanText(X_train_all)
    X_test = cleanText(X_test)

    #unique word count is calculated after the text is clean of punctuation
    X_train_tabfeatures['relative_unique_word_count'] = X_train_all[0].apply(
        lambda text: (len(set(w for w in text.split())) / len(text.split())))
    X_test_tabfeatures['relative_unique_word_count'] = X_test[0].apply(
        lambda text: (len(set(w for w in text.split())) / len(text.split())))

    tabular_features_train = normalize(np.array(X_train_tabfeatures))
    tabular_features_test = normalize(np.array(X_test_tabfeatures))

    text_train = np.array(X_train_all[0])
    text_test = np.array(X_test[0])

    check_balance(X_train_all, y_train_all)

    vectorizer_word, vectorizer_char = get_vectorizers(text_train)

    train_features_word = vectorizer_word.transform(np.array(text_train))
    test_features_word = vectorizer_word.transform(np.array(text_test))

    train_features_char = vectorizer_char.transform(np.array(text_train))
    test_features_char = vectorizer_char.transform(np.array(text_test))

    train_features = hstack([tabular_features_train, train_features_char, train_features_word]).tocsr()
    test_features = hstack([tabular_features_test, test_features_char, test_features_word]).tocsr()

    model = Classification(train_features, y_train_all, 666, params)
    cross_validate(model, train_features, y_train_all)
    test_model(model, test_features, y_test)

    return model

#creating dataframes for training and test data to iterate over with .apply() later
def read_data(path):
  res_data = []
  print('Read data...')

  if len(os.listdir(path)):
    for files in os.listdir(path):
      print('>>>',files)
      with open(os.path.join(path,files),'r', encoding='utf-8') as f:
        res_data.append(f.read())
  return res_data

#splitting texts and assigning labels, transforming dataframes into
def split_text(text, subtext_length, label):
    cursor = 0
    subtexts = []
    labels = []
    textsplit = text.split()
    for i in range(9999999999999):
        if len(textsplit) - cursor > subtext_length:
            subtexts.append(' '.join(textsplit[cursor:cursor+subtext_length]))
            labels.append(label)
            cursor += subtext_length
        else:
            subtexts.append(' '.join(textsplit[cursor:]))
            labels.append(label)
            break
    return subtexts, labels

def get_tab_features(data):
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

#cleaning texts - removing special characters, punctuation, html tags, etc.
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

#checking dataset balance
def check_balance(X_train_all, y_train_all):
    hem = 0
    other = 0

    for i in range(len(X_train_all)):
        if y_train_all[i] == 0:
          other += 1
        else:
          hem +=1
    print(f'Hem: {hem}, Other: {other}')

def tokenizer(text):
    text = bpemb_ru.encode(text)
    return text

#vectorizing texts with tf/idf
def get_vectorizers(text_train):
    vectorizer_word = TfidfVectorizer(
                                 max_features = 50000,
                                 strip_accents = 'unicode',
                                 tokenizer = tokenizer,
                                 analyzer = 'word',
                                 ngram_range = (1,3),
                                 norm = 'l2',
                                 use_idf = True,
                                 smooth_idf = True,
                                 #sublinear_tf = True
                                 min_df = 5,
                                 max_df = 95
                                )

    vectorizer_word.fit(text_train)

    vectorizer_char = TfidfVectorizer(
                                 max_features = 50000,
                                 strip_accents = 'unicode',
                                 analyzer = 'char',
                                 ngram_range = (2,3),
                                 norm = 'l2',
                                 use_idf = True,
                                 smooth_idf = True,
                                 #sublinear_tf = True
                            )
    vectorizer_char.fit(text_train)
    pickle.dump(vectorizer_word, open("vectorizer_word.pickle", "wb"))
    pickle.dump(vectorizer_char, open("vectorizer_char.pickle", "wb"))
    with open("vectorizer_word.pickle", "rb") as wordvec:
        vectorizer_word = pickle.load(wordvec)
    with open("vectorizer_word.pickle", "rb") as charvec:
        vectorizer_char = pickle.load(charvec)
    return vectorizer_word, vectorizer_char


def Classification(xtrain, ytrain, random_seed, params):
    evals_result = {}
    train_matrix, valid_matrix, y_train, y_valid = train_test_split(xtrain, ytrain, test_size = 0.2, random_state = random_seed)

    d_train = lightgbm.Dataset(train_matrix, label = y_train)
    d_valid = lightgbm.Dataset(valid_matrix, label = y_valid)

    valid = [d_train, d_valid]

    lgbmc = lightgbm.train(params = params,
                           train_set = d_train,
                           valid_sets = valid,
                           verbose_eval = 50,
                           num_boost_round = 10000000,
                           early_stopping_rounds = 2000,
                           evals_result = evals_result,
                           )

    lgbmc.save_model('lightgbm_model.txt')
    lgbmc = lightgbm.Booster(model_file='lightgbm_model.txt')

    return lgbmc

#custom evaluation metric to be used for cross validation later as lightgbm doesn't have f1 score 'from the box'
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat < 0.5, 0, 1)
    return 'f1', f1_score(y_true, y_hat), True

#training the model
#apparenty not using lambdas or using very low values gives better results
params = {
          'learning_rate': 0.008,
          'application': 'binary',
          'path_smooth' : 4,
          'max_bin' : 2,
          'max_depth': 1,
          'num_leaves': 2,
          'feature_fraction': 0.0003,
          'verbosity': -1,
          'n_thread': -1,
          'min_data_in_leaf' : 0,
          'metric': 'xentropy',
          'lambda_l1': 0.0,
          'lambda_l2': 0.0
          }

#cross validation evaluation
def cross_validate(model, train_features, y_train_all):
    cv_dataset = lightgbm.Dataset(train_features, label = y_train_all)

    cv = lightgbm.cv(
        init_model=model,
        params=params,
        #eval_train_metric = True,
        metrics = 'xentropy',
        feval = lgb_f1_score,
        nfold = 5,
        stratified = True,
        train_set=cv_dataset,
        num_boost_round=100000000,
        early_stopping_rounds=2000,
        verbose_eval=50,
        shuffle=True
    )

def test_model(model, x, y):
    y_pred = np.round(model.predict(x))
    print(classification_report(y_pred, y, labels = [1,0], target_names = ['Hem', 'Other']))


model = train_model(train_data_path, test_data_path)
