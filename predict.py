import lightgbm
import feature_extractor
import pandas as pd
import models

class Predictor():
    def __init__(self, text, sentence_amount = 0, step = 1):
        self.text = text
        self.sentence_amount = sentence_amount
        self.step = step
        self.extractor = feature_extractor.Feature_extractor()
        self.modules = models.Model_modules()

    def predict_author(self):
        features = self.extractor.featurize(pd.DataFrame([self.text]), self.sentence_amount, is_train_data = False, keep_labels = False)
        y_pred = self.modules.model.predict(features)
        mean_pred = sum(y_pred)/len(y_pred)
        return mean_pred
