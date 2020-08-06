import os
import re
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from bpemb import BPEmb
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import models

#featurizing text passed to the method as pd.DataFrame
#returns features for prediction or features and labels for training
class Feature_exctractor():
    def __init__(self):
        self.eps = 0.000000000000000001 #to prevent division by 0
        self.modules = models.Model_modules()

    def featurize(self, text_df, sentense_count = 5, is_train_data = False, keep_labels = False):
        temp_df = pd.DataFrame()
        labels_l = []
        for i in range(len(text_df[0])):
            df, df_labels = self.split_text(text_df[0][i], sentence_amount=5, label=i)
            temp_df = temp_df.append(df)
            labels_l = labels_l + df_labels            
        text_df, labels = temp_df, labels_l
        
        tabular_features = self.get_tabular_features(text_df)
        text_df = self.clean_text(text_df)
        # unique word count is calculated after the text is clean of punctuation
        tabular_features['relative_unique_word_count'] = text_df[0].apply(
            lambda text: (len(set(w for w in text.split())) / (self.eps + len(text.split()))))
        tabular_features = self.normalize(np.array(tabular_features))
        #removing stopwords
        text_df[0] = text_df[0].apply(lambda x: ' '.join([item for item in x.split() if item not in self.modules.sw]))
        
        text_array = np.array(text_df[0])

        if is_train_data:
            #if training - call get_vectorizers which will fit vectorizers and save
            #new vectorizers as pickle files
            self.modules.get_vectorizers(text_array)
            #then renew vectorizers by loading an instance of 'Model_modules',
            #which will get freshly fitted vectorizers unpickled in it's constructor
            self.modules = models.Model_modules()

        features_word = self.modules.vectorizer_word.transform(np.array(text_array))
        features_char = self.modules.vectorizer_char.transform(np.array(text_array))
        features = hstack([tabular_features, features_char, features_word]).tocsr()
        
        if keep_labels:
            return features, labels
        else:
            return features

    #splitting texts depending on the chosen sentence amount per sample
    #returns pd.DataFrame of subtexts and labels
    def split_text(self, text, sentence_amount = 5, label = 0):
        cursor = 0
        subtexts = []
        labels = []
        sentenses = ''
        textsplit = re.findall('(.*?(?:\?|\!|\.))', text)
        for subtext in textsplit:
            sentenses = sentenses + subtext
            if (cursor != 0) and (cursor % sentence_amount == 0):
              subtexts.append(sentenses)
              labels.append(label)
              sentenses = ''
            cursor += 1
        return pd.DataFrame(subtexts), labels

    def get_tabular_features(self, data):
        tabfeatures = pd.DataFrame()
        #text's total length
        tabfeatures['total_length'] = data[0].apply(len)
        #exclamation mark count
        tabfeatures['exclamation_count'] = data[0].apply(lambda text: text.count('!')/(self.eps+len(text)))
        #quotes count
        tabfeatures['quote_count'] = data[0].apply(lambda text: sum(text.count(q) for q in '"«»')/(self.eps+len(text)))
        #question mark count
        tabfeatures['question_count'] = data[0].apply(lambda text: text.count('?')/(self.eps+len(text)))
        #punctuation count
        tabfeatures['punctuation_count'] = data[0].apply(lambda text: sum(text.count(w) for w in '.,;:')/(self.eps+len(text)))
        #amount of upper case letters
        tabfeatures['uppercase_amount'] = data[0].apply(lambda text: sum(1 for c in text if c.isupper())/(self.eps+len(text)))
        #amount of upper case letters compared to the text's length
        tabfeatures['FULLCAPS_COUNT'] = tabfeatures['uppercase_amount']/(self.eps+tabfeatures['total_length'])
        #amount of unique words compared to total word count
        tabfeatures['sentence_length'] = data[0].apply(lambda text: len(text)/(self.eps+len(re.findall('(.*?(?:\?|\!|\.))', text))))
        #sentence length
        tabfeatures['word_per_length'] = data[0].apply(lambda text: len(text)/(self.eps+len(text.split())))
        #word amount per length
        tabfeatures['word_per_sentence'] = data[0].apply(lambda text: len(text.split())/(self.eps+len(re.findall('(.*?(?:\?|\!|\.))', text))))
        #word amount per sentence
        tabfeatures['word_length'] = data[0].apply(lambda text: sum(len(word) for word in text.split())/(self.eps+len(text.split())))
        #word average length
        return tabfeatures

    #cleaning texts - removing special characters, punctuation, html tags, etc.
    def clean_punc(self, text):
        no_punctuation = re.sub(r'[?|!|\'|"|#]',r'',text)
        no_punctuation = re.sub(r'[.|,|)|(|\|/|—]',r' ',text)
        no_punctuation = no_punctuation.strip()
        no_punctuation = no_punctuation.replace("\n"," ")
        return no_punctuation

    def clean_html(self, text):
        cleanr = re.compile('<.*?>')
        no_html = re.sub(cleanr, ' ', str(text))
        return no_html

    def keep_alpha(self, text):
        alpha_sent = ""
        for word in text.split():
            alpha_word = re.sub('[^а-я А-Я]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent

    def clean_text(self, data):
        data[0] = data[0].str.lower()
        data[0] = data[0].apply(self.clean_html)
        data[0] = data[0].apply(self.clean_punc)
        data[0] = data[0].apply(self.keep_alpha)
        return data

    def normalize(self, array):
      norm = np.linalg.norm(array)
      array = array/norm
      return array
