import lightgbm
import feature_extractor
import pandas as pd
import models

class Predictor():
    def __init__(self, text):
        self.text = text
        self.extractor = feature_extractor.Feature_exctractor()
        self.modules = models.Model_modules()

    def predict_author(self, sentense_count = 5):
        features = self.extractor.featurize(pd.DataFrame([self.text]), sentense_count = 5, is_train_data = False, keep_labels = False)
        y_pred = self.modules.model.predict(features)
        pred_average = sum(y_pred)/len(y_pred)
        return pred_average
