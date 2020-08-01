#!/usr/bin/env python

#tabfeatures
# def import_text(text):
#     res_data = []
#     print('Read data...')
#
#     if len(os.listdir(path)):
#         for files in os.listdir(path):
#             print('>>>', files)
#             with open(os.path.join(path, files), 'r') as f:
#                 res_data.append(f.read())
#     return res_data
#
# train_data = import_text(text_path)
#
# def split_text(text, subtext_length=300):
#     cursor = 0
#     subtexts = []
#     textsplit = text.split()
#     for i in range(9999999999999):
#         if len(textsplit) - cursor > subtext_length:
#             subtexts.append(' '.join(textsplit[cursor:cursor+subtext_length]))
#             cursor += subtext_length
#         else:
#             subtexts.append(' '.join(textsplit[cursor:]))
#             break
#
#     return subtexts
#
# X_test = split_text(train_data[0][0], 300)
#
# def PrepareText(data):
#     tabfeatures = pd.DataFrame()
#     #Text's total length
#     tabfeatures['total_length_count'] = data[0].apply(len)
#     #Exclamation mark count
#     tabfeatures['exclamation_count'] = data[0].apply(lambda text: text.count('!'))
#     #Question mark count
#     tabfeatures['question_count'] = data[0].apply(lambda text: text.count('?'))
#     #Punctuation count
#     tabfeatures['punctuation_count'] = data[0].apply(lambda text: sum(text.count(w) for w in '.,;:'))
#     #Amount of upper case letters
#     tabfeatures['uppercase_amount'] = data[0].apply(lambda text: sum(1 for c in text if c.isupper()))
#     #Amount of upper case letters compared to the text's length
#     tabfeatures['FULLCAPS_COUNT'] = tabfeatures['uppercase_amount']/tabfeatures['total_length_count']
#     #Amount of unique words compared to total word count
#
#     return tabfeatures
#
# X_test_tabfeatures = PrepareText(X_test)
#
# #formatting
# def cleanPunc(text):
#     no_punctuation = re.sub(r'[?|!|\'|"|#]',r'',text)
#     no_punctuation = re.sub(r'[.|,|)|(|\|/|—]',r' ',text)
#     no_punctuation = no_punctuation.strip()
#     no_punctuation = no_punctuation.replace("\n"," ")
#     return no_punctuation
#
# def cleanHtml(text):
#     cleanr = re.compile('<.*?>')
#     no_html = re.sub(cleanr, ' ', str(text))
#     return no_html
#
# def keepAlpha(text):
#     alpha_sent = ""
#     for word in text.split():
#         alpha_word = re.sub('[^а-я А-Я]+', ' ', word)
#         alpha_sent += alpha_word
#         alpha_sent += " "
#     alpha_sent = alpha_sent.strip()
#     return alpha_sent
#
# def cleanText(data):
#     data[0] = data[0].str.lower()
#     data[0] = data[0].apply(cleanHtml)
#     data[0] = data[0].apply(cleanPunc)
#     data[0] = data[0].apply(keepAlpha)
#
#     return data
#
# X_test = cleanText(X_test)
#
# #getting amount of unique words per text
# X_test_tabfeatures['relative_unique_word_count'] = X_test[0].apply(lambda text: (len(set(w for w in text.split()))/len(text.split())))
#
# def normalize(array):
#   norm = np.linalg.norm(array)
#   array = array/norm
#   return array
#
# tabular_features_test = normalize(np.array(X_train_tabfeatures))
#
# bpemb_ru = BPEmb(lang='ru', vs=100000)
# def tokenizer(text):
#     text = bpemb_ru.encode(text)
#     return text
#
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# with open("vectorizer_word.pickle", "rb") as wordvec:
#      vectorizer_word = pickle.load(wordvec)
#
# with open("vectorizer_word.pickle", "rb") as charvec:
#     vectorizer_char = pickle.load(charvec)
#
# from scipy.sparse import hstack
#
# test_features_word = vectorizer_word.transform(X_test)
# test_features_char = vectorizer_char.transform(X_test)
#
# test_features = hstack([tabular_features_test, test_features_char, test_features_word]).tocsr()

#load model
with open('model.pkl', 'rb') as fin:
    model = pickle.load(fin)

def predict(model, text):
    y_pred = np.round(model.predict(x))
    return y_pred

def main():
    text = '''Эта книга адресована всем, кто изучает русский язык. Но состоит она не из правил, упражнений и учебных текстов. Для этого созданы другие замечательные учебники.
У этой книги совсем иная задача. Она поможет вам научиться не только разговаривать, но и размышлять по-русски. Книга, которую вы держите в руках, составлена из афоризмов и размышлений великих мыслителей, писателей, поэтов, философов и общественных деятелей различных эпох. Их мысли - о тех вопросах, которые не перестают волновать человечество.
Вы можете соглашаться или не соглашаться с тем, что прочитаете в этой книге. Возможно, вам покажется, что какие-то мысли уже устарели. Но вы должны обязательно подумать и обосновать, почему вы так считаете.'''
    result = predict(model, text)
    print('Скрипт действительно сработал?')
    print(result)

if __name__ == "__main__":
    print("allahu akbar")
    main()