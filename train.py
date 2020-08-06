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
from nltk.corpus import stopwords
import feature_extractor

class Trainer():
    def __init__(self):
        self.train_data_path = r'./data/train'
        self.test_data_path = r'./data/test'
        self.extractor = feature_extractor.Feature_exctractor()

        #apparenty not using lambdas or using very low values gives better results
        self.params = {
                  'learning_rate': 0.07,
                  'application': 'binary',
                  'path_smooth' : 0,
                  'max_bin' : 2,
                  'max_depth': 1,
                  'num_leaves': 2,
                  'feature_fraction': 0.00003,
                  'verbosity': -1,
                  'n_thread': -1,
                  'min_data_in_leaf' : 0,
                  'metric': 'xentropy',
                  'lambda_l1': 0.0,
                  'lambda_l2': 0.0
                  }

    def train_model(self, sentense_count = 5):
        train_texts = self.read_data(self.train_data_path)
        test_texts = self.read_data(self.test_data_path)

        train_features, train_labels = self.extractor.featurize(train_texts, sentense_count, is_train_data = True, keep_labels = True)
        test_features, test_labels = self.extractor.featurize(test_texts, sentense_count, is_train_data = False, keep_labels = True)

        print(len(train_labels))
        self.train_clf(train_features, train_labels, 666, self.params)
        self.model = lightgbm.Booster(model_file='lightgbm_model.txt')
        self.cross_validate(self.model, train_features, train_labels)
        self.test_model(self.model, test_features, test_labels)
        return "Training complete"


    def read_data(self, path):
        texts = []
        print(f'Reading data in {path}...')

        if len(os.listdir(path)):
            for files in os.listdir(path):
                print('>>>', files)
                with open(os.path.join(path, files), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        return pd.DataFrame(texts)

    def train_clf(self, train_features, train_labels, random_seed, params):
        print('Training...')
        evals_result = {}
        train_matrix, valid_matrix, y_train, y_valid = train_test_split(train_features, train_labels, test_size=0.2, random_state=random_seed)

        d_train = lightgbm.Dataset(train_matrix, label=y_train)
        d_valid = lightgbm.Dataset(valid_matrix, label=y_valid)
        valid = [d_train, d_valid]

        lgbmc = lightgbm.train(params=params,
                               train_set=d_train,
                               valid_sets=valid,
                               verbose_eval=500,
                               num_boost_round=10000000,
                               early_stopping_rounds=2000,
                               evals_result=evals_result,
                               )
        lgbmc.save_model('lightgbm_model.txt')
        print('Training complete!')

    #custom evaluation metric to be used for cross validation later as lightgbm doesn't have f1 score 'from the box'
    def lgb_f1_score(self, y_hat, data):
        y_true = data.get_label()
        y_hat = np.where(y_hat < 0.5, 0, 1)
        return 'f1', f1_score(y_true, y_hat), True

    #cross validation evaluation
    def cross_validate(self, model, train_features, y_train_all):
        print('Cross validating...')
        cv_dataset = lightgbm.Dataset(train_features, label = y_train_all)

        cv = lightgbm.cv(
            init_model=model,
            params=self.params,
            #eval_train_metric = True,
            metrics = 'xentropy',
            feval = self.lgb_f1_score,
            nfold = 5,
            stratified = True,
            train_set=cv_dataset,
            num_boost_round=100000000,
            early_stopping_rounds=100,
            verbose_eval=50,
            shuffle=True
        )
        print('Cross-validation complete')

    def test_model(self, model, x, y):
        y_pred = np.round(model.predict(x))
        return print(classification_report(y_pred, y, labels = [1,0], target_names = ['Hem', 'Other']))
