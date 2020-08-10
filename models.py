import pickle
import lightgbm
from bpemb import BPEmb
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import custom_unpickler

#importing models
class Model_modules():
    def __init__(self):
        self.sw = set(stopwords.words('russian'))
        try:
            with open("vectorizer_word.pickle", "rb") as wordvec:
                unpickler = custom_unpickler.MyCustomUnpickler(wordvec)
                self.vectorizer_word = unpickler.load()
            with open("vectorizer_char.pickle", "rb") as charvec:
                self.vectorizer_char = pickle.load(charvec)
            self.model = lightgbm.Booster(model_file='lightgbm_model.txt')
        except Exception:
            print('models are none')
            self.vectorizer_word = None
            self.vectorizer_char = None
            self.model = None

    def get_vectorizers(self, text):
        print('Fitting vectorizers...')
        self.vectorizer_word = TfidfVectorizer(
            max_features=50000,
            strip_accents='unicode',
            tokenizer=tokenizer,
            analyzer='word',
            ngram_range=(1, 3),
            norm='l2',
            #use_idf=True,
            #smooth_idf=True,
            sublinear_tf=True,
            min_df=2,
            max_df=98
        )

        self.vectorizer_word.fit(text)

        self.vectorizer_char = TfidfVectorizer(
            max_features=50000,
            strip_accents='unicode',
            analyzer='char',
            ngram_range=(1, 6),
            norm='l2',
            #use_idf=True,
            #smooth_idf=True,
            sublinear_tf=True
        )

        self.vectorizer_char.fit(text)

        pickle.dump(self.vectorizer_word, open("vectorizer_word.pickle", "wb"))
        pickle.dump(self.vectorizer_char, open("vectorizer_char.pickle", "wb"))

bpemb_ru = BPEmb(lang='ru', vs=10000)
def tokenizer(text):
    text = bpemb_ru.encode(text)
    return text
