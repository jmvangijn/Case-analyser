import pandas as pd
import re
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer


#nltk.download()


def convert_case(case):
#converting
        sentence = ', '.join(case)
        letters_only = re.sub("[^a-zA-Z]", " ", sentence)
        lower_case = letters_only.lower()
        words = lower_case.split()
        #nostop_words = [w for w in words if not w in stopwords.words("dutch")]
        return( " ".join( words ))


def import_caseset():
    train = pd.read_csv("notes_fulset.txt", header=0, delimiter=",", quoting=3)
    print("Import succesfull!")
    print(train.columns.values)
    print(train.shape)
    pre_analysis_set = []
    case_set = train.values
    for case in case_set:
        word_set = convert_case(case)
        pre_analysis_set.append(word_set)
    print("Preprocessing done!")
    return pre_analysis_set


def vectorize_that_shit():
    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=50000)
    train_data_features = vectorizer.fit_transform(import_caseset())
    feature_set = train_data_features.toarray()
    print("Finished vectors!")
    print(feature_set.shape)
    print(feature_set)
    #dist = train_data_features.sum()
    vocab = vectorizer.get_feature_names()
    return feature_set, vocab


if __name__ == '__main__':
    vectorize_that_shit()
