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
class Feature_extractor():
    def __init__(self):
        self.eps = 0.000000000000000001 #to prevent division by 0
        self.modules = models.Model_modules()

    def featurize(self, text_df, sentence_amount = 5, step = 1, is_train_data = False, keep_labels = False):
        temp_df = pd.DataFrame()
        labels_l = []
        #if sentence_amount is set to 0, corpus is not splitted into subtexts
        #this is only used for making predictions so there is no need to assign labels
        if sentence_amount == 0:
            pass
        else:            
            for i in range(len(text_df[0])):
                df, df_labels = self.split_text(text_df[0][i], sentence_amount=sentence_amount, step=step, label=i)
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
        try:
            features_word = self.modules.vectorizer_word.transform(np.array(text_array))
            features_char = self.modules.vectorizer_char.transform(np.array(text_array))
            features = hstack([tabular_features, features_char, features_word]).tocsr()
        except:
             print('Vectorizer modules have not been trained yet. Train the predictor first.')
             return
        
        if keep_labels:
            return features, labels
        else:
            return features

    #splitting texts depending on the chosen sentence amount per sample
    #returns pd.DataFrame of subtexts and labels
    def split_text(self, text, sentence_amount = 5, step = 0, label = 0):
        cursor = 0
        subtexts = []
        labels = []
        sentences = ''
        #this regex pattern finds all sentences that end with small letter + ./!/?
        #to prevent text parts like "John F. Kennedy" splitting the corpus in wrong places,
        #mainaining accumulation of consistent and complete sentences
        textsplit = re.findall('(.*?(?:[а-я]\?|[а-я]\!|[а-я]\.))', text)
        for subtext in textsplit:
            sentences = sentences + subtext
            if step == 0:
                cursor += 1
            else:
                cursor += step
            if (cursor != 0) and (cursor % sentence_amount == 0):
              subtexts.append(sentences)
              labels.append(label)
              sentences = ''
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
        #line count
        tabfeatures['question_count'] = data[0].apply(lambda text: text.count('–')/(self.eps+len(text)))
        #punctuation count
        tabfeatures['punctuation_count'] = data[0].apply(lambda text: sum(text.count(w) for w in '.,;:')/(self.eps+len(text)))
        #amount of upper case letters
        tabfeatures['uppercase_amount'] = data[0].apply(lambda text: sum(1 for c in text if c.isupper())/(self.eps+len(text)))
        #amount of upper case letters compared to the text's length
        tabfeatures['FULLCAPS_COUNT'] = tabfeatures['uppercase_amount']/(self.eps+tabfeatures['total_length'])
        #amount of unique words compared to total word count
        tabfeatures['sentence_length'] = data[0].apply(lambda text: len(text)/(self.eps+len(re.findall('(.*?(?:[а-я]\?|[а-я]\!|[а-я]\.))', text))))
        #sentence length
        tabfeatures['word_per_length'] = data[0].apply(lambda text: len(text)/(self.eps+len(text.split())))
        #word amount per length
        tabfeatures['word_per_sentence'] = data[0].apply(lambda text: len(text.split())/(self.eps+len(re.findall('(.*?(?:[а-я]\?|[а-я]\!|[а-я]\.))', text))))
        #word amount per sentence
        tabfeatures['word_length'] = data[0].apply(lambda text: sum(len(word) for word in text.split())/(self.eps+len(text.split())))
        #total length is not included in the tabular features to prevent
        #the model from following the easiest prediction pattern in case
        #total length will be a determining factor, especially when predicting
        #on an unsplitted text corpus
        tabfeatures.drop(['total_length'], axis=1)
        #word average length
        return tabfeatures

    #cleaning texts - removing special characters, punctuation, html tags, etc.
    def clean_punc(self, text):
        no_punctuation = re.sub(r'[?|!|\'|"|#|_|«|»]',r'',text)
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
