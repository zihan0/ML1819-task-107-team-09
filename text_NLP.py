import nltk
from nltk.corpus import stopwords
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
import string
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def find_features(top_1000_words, text):
    feature = {}
    for word in top_1000_words:
        feature[word] = word in text.lower()
    return feature

df = pd.read_csv('Dataset/gender-classifier-DFE-791531.csv', encoding='latin1')
df = shuffle(shuffle(shuffle(df)))

all_descriptions = df['description']
all_tweets = df['text']
all_genders = df['gender']
all_gender_confidence = df['gender:confidence']
description_tweet_gender = []

bag_of_words = []
c = 0
stop = stopwords.words('english')
for tweet in all_tweets:
    description = all_descriptions[c]
    gender = all_genders[c]
    gender_confidence = all_gender_confidence[c]
    if (str(tweet) == 'nan' and str(description) == 'nan') or str(gender) == 'nan' or str(gender) == 'unknown' or float(gender_confidence) < 0.8:
        c += 1
        continue

    if str(tweet) == 'nan':
        tweet = ''
    if str(description) == 'nan':
        description = ''

    # removal of punctuations
    for punct in string.punctuation:
        if punct in tweet:
            tweet = tweet.replace(punct, " ")
        if punct in description:
            description = description.replace(punct, " ")

    # adding the word to the bag
    for word in tweet.split():
        if word.isalpha() and word.lower() not in stop:
            bag_of_words.append(word.lower())
    for word in description.split():
        if word.isalpha() and word.lower() not in stop:
            bag_of_words.append(word.lower())

    description_tweet_gender.append((tweet + " " + description, gender))
    c += 1

bag_of_words = nltk.FreqDist(bag_of_words)
top_1000_words = []
for word in bag_of_words.most_common(20):
    top_1000_words.append(word[0])

top_1000_words

feature_set = [(find_features(top_1000_words, text), gender) for (text, gender) in description_tweet_gender]
training_set = feature_set[:int(len(feature_set) * 3 / 4)]
testing_set = feature_set[int(len(feature_set) * 3 / 4):]

print("Length of feature set", len(feature_set))
print("Length of training set", len(training_set))
print("Length of testing set", len(testing_set))

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
accuracy = nltk.classify.accuracy(LogisticRegression_classifier, training_set) * 100
print("Logistic Regression classifier Training Accuracy =", accuracy)
accuracy = nltk.classify.accuracy(LogisticRegression_classifier, testing_set) * 100
print("Logistic Regression classifier Test Accuracy =", accuracy)

# creating a naive bayes classifier
NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
accuracy = nltk.classify.accuracy(NB_classifier, training_set) * 100
print("Naive Bayes Classifier Training Accuracy =", accuracy)
accuracy = nltk.classify.accuracy(NB_classifier, testing_set) * 100
print("Naive Bayes Classifier Test Accuracy =", accuracy)
NB_classifier.show_most_informative_features(20)


# creating a multinomial naive bayes classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
accuracy = nltk.classify.accuracy(MNB_classifier, training_set) * 100
print("Multinomial Naive Bayes Classifier Training Accuracy =", accuracy)
accuracy = nltk.classify.accuracy(MNB_classifier, testing_set) * 100
print("Multinomial Naive Bayes Classifier Test Accuracy =", accuracy)