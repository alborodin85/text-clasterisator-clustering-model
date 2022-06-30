import sklearn.metrics

import pandas_options
import pandas as pd

dataFrame = pd.read_csv('hierarchical_text_classification/train_40k.csv')

print(dataFrame)
print()

texts = dataFrame['Text'][:10000]
labels = dataFrame['Cat1'][:10000]


def clearText(rawString: str):
    rawString = rawString.strip()
    rawString = rawString.replace('.', '')
    rawString = rawString.lower()
    return rawString


texts = texts.apply(clearText)
print(texts[1])

import unicodedata
import sys
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
from nltk.corpus import stopwords

# nltk.download('stopwords')
stop_words = stopwords.words('english')

punctuation = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')
)

from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')
# nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


def replacePunctuation(rawString: str):
    rawString = rawString.translate(punctuation)
    rawString = word_tokenize(rawString)
    rawString = [word for word in rawString if word not in stop_words]
    rawString = [lemmatizer.lemmatize(word, pos='v') for word in rawString]
    # rawString = [porter.stem(word) for word in rawString]
    rawString = ' '.join(rawString)
    return rawString


texts = texts.apply(replacePunctuation)
print(texts[1])

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(texts)

print(feature_matrix.toarray()[1])
print(tfidf.vocabulary_)
print(feature_matrix.toarray().shape)

from sklearn_som.som import SOM
from sklearn import datasets
from Estimator import Estimator

n_clusters = 6
train = feature_matrix.toarray()

labelsUniq = list(set(labels))

labelVector = []

for labelFactId in range(len(labels)):
    for labelNameId in range(len(labelsUniq)):
        if labels[labelFactId] == labelsUniq[labelNameId]:
            labelVector.append(labelNameId)
            break

print(labels)
print(labelVector)

targets = labelVector

iris_som = SOM(m=n_clusters, n=1, dim=feature_matrix.toarray().shape[1])

iris_som.fit(train, epochs=1)
predictions = iris_som.predict(train)
print('targets')
print(targets)

print('predictions')
print(predictions)

estimation = Estimator.estimate(n_clusters, predictions, targets, train)

print('estimation')
print(estimation[0])
