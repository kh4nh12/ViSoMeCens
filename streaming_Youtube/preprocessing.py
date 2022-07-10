import pandas as pd
import numpy as np
from scipy import sparse
import regex as re
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from pyvi import ViTokenizer, ViPosTagger, ViUtils
import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
import os


"""check repeated character in word"""


def check_repeated_character(text):
    text = re.sub('  +', ' ', text).strip()
    count = {}
    for i in range(len(text) - 1):
        if text[i] == text[i + 1]:
            return True
    return False


def check_space(text):  # check space in string
    for i in range(len(text)):
        if text[i] == ' ':
            return True
    return False


def check_special_character_numberic(text):
    return any(not c.isalpha() for c in text)


# Remove emoji and emoticons
def remove_emoji(text):
    for emot in UNICODE_EMOJI:
        text = str(text).replace(emot, ' ')
    text = re.sub('  +', ' ', text).strip()
    return text


# Remove url
def url(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' ', str(text))
    text = re.sub('  +', ' ', text).strip()
    return text


# remove special character
def special_character(text):
    text = re.sub(r'\d+', lambda m: " ", text)
    # text = re.sub(r'\b(\w+)\s+\1\b',' ', text) #remove duplicate number word
    text = re.sub("[~!@#$%^&*()_+{}“”|:\"<>?`´\-=[\]\;\\\/.,]", " ", text)
    text = re.sub('  +', ' ', text).strip()
    return text


# normalize repeated characters
def repeated_character(text):
    text = re.sub(r'(\w)\1+', r'\1', text)
    text = re.sub('  +', ' ', text).strip()
    return text


def mail(text):
    text = re.sub(r'[^@]+@[^@]+\.[^@]+', ' ', text)
    text = re.sub('  +', ' ', text).strip()
    return text


# remove mention tag and hashtag
def tag(text):
    text = re.sub(r"(?:\@|\#|\://)\S+", " ", text)
    text = re.sub('  +', ' ', text).strip()
    return text


# """Remove all mixed words and numbers"""
def mixed_word_number(text):
    text = ' '.join(s for s in text.split() if not any(c.isdigit() for c in s))
    text = re.sub('  +', ' ', text).strip()
    return text


# tokenize by lib Pyvi
def tokenize(text):
    text = str(text)
    text = ViTokenizer.tokenize(text)
    return text


""" emoji """
c2e_path = os.path.join(os.getcwd(), 'dictionary/character2emoji.xlsx')
character2emoji = pd.read_excel(c2e_path)  # character to emoji


def convert_character2emoji(text):
    text = str(text)
    for i in range(character2emoji.shape[0]):
        text = text.replace(character2emoji.at[i, 'character'], " " + character2emoji.at[i, 'emoji'] + " ")
    text = re.sub('  +', ' ', text).strip()
    return text


e2w_path = os.path.join(os.getcwd(), 'dictionary/emoji2word.xlsx')
emoji2word = pd.read_excel(e2w_path)  # emoji to word


def convert_emoji2word(text):
    for i in range(emoji2word.shape[0]):
        text = text.replace(emoji2word.at[i, 'emoji'], " " + emoji2word.at[i, 'word_vn'] + " ")
    text = re.sub('  +', ' ', text).strip()
    return text


""" abbreviation """
adn_path = os.path.join(os.getcwd(), 'dictionary/abb_dict_normal.xlsx')
abb_dict_normal = pd.read_excel(adn_path)


def abbreviation_normal(text):  # len word equal 1
    text = str(text)
    temp = ''
    for word in text.split():
        for i in range(abb_dict_normal.shape[0]):
            if str(abb_dict_normal.at[i, 'abbreviation']) == str(word):
                word = str(abb_dict_normal.at[i, 'meaning'])
        temp = temp + ' ' + word
    text = temp
    text = re.sub('  +', ' ', text).strip()
    return text


ads_path = os.path.join(os.getcwd(), 'dictionary/abb_dict_special.xlsx')
abb_dict_special = pd.read_excel(ads_path)


def abbreviation_special(text):  # including special character and number
    text = ' ' + str(text) + ' '
    for i in range(abb_dict_special.shape[0]):
        text = text.replace(' ' + abb_dict_special.at[i, 'abbreviation'] + ' ',
                            ' ' + abb_dict_special.at[i, 'meaning'] + ' ')
    text = re.sub('  +', ' ', text).strip()
    return text


def special_character_1(text):  # remove dot and comma
    text = re.sub("[.,?!]", " ", text)
    text = re.sub('  +', ' ', text).strip()
    return text


def abbreviation_kk(text):
    text = str(text)
    for t in text.split():
        if 'kk' in t:
            text = text.replace(t, ' ha ha ')
        else:
            if 'kaka' in t:
                text = text.replace(t, ' ha ha ')
            else:
                if 'kiki' in t:
                    text = text.replace(t, ' ha ha ')
                else:
                    if 'haha' in t:
                        text = text.replace(t, ' ha ha ')
                    else:
                        if 'hihi' in t:
                            text = text.replace(t, ' ha ha ')
    text = re.sub('  +', ' ', text).strip()
    return text


def annotations(dataset):
    pos = []
    max_len = 8000
    for i in range(dataset.shape[0]):
        n = len(dataset.at[i, 'cmt'])
        l = [0] * max_len
        s = int(dataset.at[i, 'start_index'])
        e = int(dataset.at[i, 'end_index'])
        for j in range(s, e):
            l[j] = 1
        pos.append(l)
    return pos


def abbreviation_predict(t):
    model_path = os.path.join(os.getcwd(), 'model/abb_model.sav')
    loaded_model = joblib.load(model_path)

    da_path = os.path.join(os.getcwd(), 'dictionary/abbreviation_dictionary_vn.xlsx')
    train_path = os.path.join(os.getcwd(), 'dictionary/train_duplicate_abb_data.xlsx')
    dev_path = os.path.join(os.getcwd(), 'dictionary/dev_duplicate_abb_data.xlsx')
    test_path = os.path.join(os.getcwd(), 'dictionary/test_duplicate_abb_data.xlsx')
    duplicate_abb = pd.read_excel(da_path, sheet_name='duplicate', header=None)
    duplicate_abb = list(duplicate_abb[0])

    train_duplicate_abb_data = pd.read_excel(train_path)
    dev_duplicate_abb_data = pd.read_excel(dev_path)
    test_duplicate_abb_data = pd.read_excel(test_path)
    duplicate_abb_data = pd.concat([train_duplicate_abb_data, dev_duplicate_abb_data, test_duplicate_abb_data],
                                   ignore_index=True)
    duplicate_abb_data = duplicate_abb_data.drop_duplicates(keep='last').reset_index(drop=True)

    X = duplicate_abb_data[['abb', 'start_index', 'end_index', 'cmt']]
    y = duplicate_abb_data['origin']

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    enc = DictVectorizer()
    Tfidf_vect = TfidfVectorizer(max_features=1200)

    temp = annotations(X)
    X_pos = sparse.csr_matrix(np.asarray(temp))
    X_abb = enc.fit_transform(X[['abb']].to_dict('records'))
    X_text = Tfidf_vect.fit_transform(X['cmt'])
    X = hstack((X_abb, X_pos, X_text))

    text = str(t)
    max_len = 8000
    if len(t) > max_len:
        text = t[:max_len]

    cmt = ' ' + text + ' '
    for abb in duplicate_abb:
        start_index = 0
        count = 0
        while start_index > -1:  # start_index = -1 -> abb is not in cmt
            start_index = cmt.find(' ' + abb + ' ')  # find will return FIRST index abb in cmt
            if start_index > -1:
                end_index = start_index + len(abb)
                t = pd.DataFrame([[abb, start_index, end_index, text]],
                                 columns=['abb', 'start_index', 'end_index', 'cmt'], index=None)
                temp = annotations(t)
                X_pos = sparse.csr_matrix(np.asarray(temp))

                X_abb = enc.transform(t[['abb']].to_dict('records'))
                # print(t['cmt'])
                X_text = Tfidf_vect.transform([text])

                X = hstack((X_abb, X_pos, X_text))
                predict = loaded_model.predict(X)
                origin = le.inverse_transform(predict.argmax(axis=1))
                origin = ''.join(origin)
                text = text[:start_index + count * (len(origin) - len(abb))] + origin + text[end_index + count * (
                            len(origin) - len(abb)):]
                text = ''.join(text)
                count = count + 1
                for i in range(start_index + 1, end_index + 1):  # replace abb to space ' '
                    cmt = cmt[:i] + ' ' + cmt[i + 1:]
    return text


def preprocessing(text):
    text = text.lower()
    text = convert_character2emoji(text)
    text = url(text)
    text = mail(text)
    text = tag(text)
    text = mixed_word_number(text)
    text = special_character_1(text)  # ##remove , . ? !
    text = abbreviation_kk(text)
    text = abbreviation_special(text)
    text = convert_character2emoji(text)
    text = remove_emoji(text)
    text = repeated_character(text)
    text = special_character(text)
    text = abbreviation_normal(text)
    text = abbreviation_predict(text)
    text = tokenize(text)
    return text
