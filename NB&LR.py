import nltk
from nltk.corpus import stopwords
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.utils import shuffle
import string
import pandas as pd

def find_features(top_words, text):
    feature = {}
    for word in top_words:
        feature[word] = word in text.lower()
    return feature


def main() :

    df = pd.read_csv('Dataset/gender-classifier-DFE-791531.csv', encoding = 'latin1')
    #df = shuffle(shuffle(shuffle(df)))
    print(df.head(10))

    all_descriptions = df['description']
    all_tweets = df['text']
    all_genders = df['gender']
    all_gender_confidence = df['gender:confidence']
    description_tweet_gender = []

    # comment out if running the program for the first time in order to download stopwords file
    '''
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download() 
    '''
    # Creation of bag of words for the description
    bag_of_words = []
    c = 0  # for the index of the row
    stop = stopwords.words('english')
    for tweet in all_tweets:
        description = all_descriptions[c]
        gender = all_genders[c]
        gender_confidence = all_gender_confidence[c]

        # remove the rows which has an empty tweet and description
        # remove the rows with unknown or empty gender
        # remove the rows which have gender:confidence < 80%
        if (str(tweet) == 'nan' and str(description) == 'nan') or str(gender) == 'nan' or str(
                gender) == 'unknown' or float(gender_confidence) < 0.8:
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

        # adding the word to the bag except stopwords
        for word in tweet.split():
            if word.isalpha() and word.lower() not in stop:
                bag_of_words.append(word.lower())
        for word in description.split():
            if word.isalpha() and word.lower() not in stop:
                bag_of_words.append(word.lower())

            # using tweet and description for classification
        description_tweet_gender.append((tweet + " " + description, gender))
        c += 1

    print(len(bag_of_words))
    print(len(description_tweet_gender))

    # get top 4000 words which will act as our features of each sentence
    bag_of_words = nltk.FreqDist(bag_of_words)
    top_words = []
    for word in bag_of_words.most_common(4000):
        top_words.append(word[0])

    top_words[:10]

    # creating the feature set, training set and the testing set
    feature_set = [(find_features(top_words, text), gender) for (text, gender) in description_tweet_gender]

    logistic = []
    naive = []
    multinomial = []

    for i in range(10):
        temp = predict(feature_set)
        logistic.append(temp[0])
        naive.append(temp[1])
        multinomial.append(temp[2])

    print("\nFinal logistic: ", logistic)
    print("\nFinal bayes: ", naive)
    print("\nFinal multinomial: ", multinomial)

def predict(feature_set):
    # 75/25
    frac = 0.25
    inds = set(random.sample(list(range(len(feature_set))), int(frac * len(feature_set))))

    training_set1 = [n for i,n in enumerate(feature_set) if i not in inds]
    testing_set1 = [n for i,n in enumerate(feature_set) if i in inds]

    # 80/20
    frac = 0.20
    inds = set(random.sample(list(range(len(feature_set))), int(frac * len(feature_set))))
    training_set2 = [n for i,n in enumerate(feature_set) if i not in inds]
    testing_set2 = [n for i,n in enumerate(feature_set) if i in inds]

    # 70/30
    frac = 0.30
    inds = set(random.sample(list(range(len(feature_set))), int(frac * len(feature_set))))
    training_set3 = [n for i,n in enumerate(feature_set) if i not in inds]
    testing_set3 = [n for i,n in enumerate(feature_set) if i in inds]

    logistic_regression = []
    naive_regression = []
    multinomial_regression = []

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())

    LogisticRegression_classifier.train(training_set1)
    accuracy = nltk.classify.accuracy(LogisticRegression_classifier, training_set1) * 100
    logistic_regression.append(accuracy)
    print("Logistic Regression classifier Training Accuracy1 =", accuracy)

    accuracy = nltk.classify.accuracy(LogisticRegression_classifier, testing_set1) * 100
    logistic_regression.append(accuracy)
    print("Logistic Regression classifier Test Accuracy1 =", accuracy)
    LogisticRegression_classifier.train(training_set2)

    accuracy = nltk.classify.accuracy(LogisticRegression_classifier, training_set2) * 100
    logistic_regression.append(accuracy)
    print("Logistic Regression classifier Training Accuracy2 =", accuracy)
    accuracy = nltk.classify.accuracy(LogisticRegression_classifier, testing_set2) * 100
    logistic_regression.append(accuracy)
    print("Logistic Regression classifier Test Accuracy2 =", accuracy)

    LogisticRegression_classifier.train(training_set3)
    accuracy = nltk.classify.accuracy(LogisticRegression_classifier, training_set3) * 100
    logistic_regression.append(accuracy)
    print("Logistic Regression classifier Training Accuracy3 =", accuracy)
    accuracy = nltk.classify.accuracy(LogisticRegression_classifier, testing_set3) * 100
    logistic_regression.append(accuracy)
    print("Logistic Regression classifier Test Accuracy3 =", accuracy)

    # creating a naive bayes classifier
    NB_classifier = nltk.NaiveBayesClassifier.train(training_set1)
    accuracy = nltk.classify.accuracy(NB_classifier, training_set1) * 100
    naive_regression.append(accuracy)
    print("Naive Bayes Classifier Training Accuracy1 =", accuracy)
    accuracy = nltk.classify.accuracy(NB_classifier, testing_set1) * 100
    naive_regression.append(accuracy)
    print("Naive Bayes Classifier Test Accuracy1 =", accuracy)

    NB_classifier = nltk.NaiveBayesClassifier.train(training_set2)
    accuracy = nltk.classify.accuracy(NB_classifier, training_set2) * 100
    naive_regression.append(accuracy)
    print("Naive Bayes Classifier Training Accuracy2 =", accuracy)
    accuracy = nltk.classify.accuracy(NB_classifier, testing_set2) * 100
    naive_regression.append(accuracy)
    print("Naive Bayes Classifier Test Accuracy2 =", accuracy)

    NB_classifier = nltk.NaiveBayesClassifier.train(training_set3)
    accuracy = nltk.classify.accuracy(NB_classifier, training_set3) * 100
    naive_regression.append(accuracy)
    print("Naive Bayes Classifier Training Accuracy3 =", accuracy)
    accuracy = nltk.classify.accuracy(NB_classifier, testing_set3) * 100
    naive_regression.append(accuracy)
    print("Naive Bayes Classifier Test Accuracy3 =", accuracy)
    # NB_classifier.show_most_informative_features(20)


    # creating a multinomial naive bayes classifier
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set1)
    accuracy = nltk.classify.accuracy(MNB_classifier, training_set1) * 100
    multinomial_regression.append(accuracy)
    print("Multinomial Naive Bayes Classifier Training Accuracy =", accuracy)
    accuracy = nltk.classify.accuracy(MNB_classifier, testing_set1) * 100
    multinomial_regression.append(accuracy)
    print("Multinomial Naive Bayes Classifier Test Accuracy =", accuracy)

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set2)
    accuracy = nltk.classify.accuracy(MNB_classifier, training_set2) * 100
    multinomial_regression.append(accuracy)
    print("Multinomial Naive Bayes Classifier Training Accuracy2 =", accuracy)
    accuracy = nltk.classify.accuracy(MNB_classifier, testing_set2) * 100
    multinomial_regression.append(accuracy)
    print("Multinomial Naive Bayes Classifier Test Accuracy2 =", accuracy)


    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set3)
    accuracy = nltk.classify.accuracy(MNB_classifier, training_set3) * 100
    multinomial_regression.append(accuracy)
    print("Multinomial Naive Bayes Classifier Training Accuracy3 =", accuracy)
    accuracy = nltk.classify.accuracy(MNB_classifier, testing_set3) * 100
    multinomial_regression.append(accuracy)
    print("Multinomial Naive Bayes Classifier Test Accuracy3 =", accuracy)

    combined = []
    combined.append(logistic_regression)
    combined.append(naive_regression)
    combined.append(multinomial_regression)

    return combined




if __name__== "__main__":
  import time
  st = time.time()
  print("Starting\n")
  main()

  print("----%.2f----" % (time.time() - st))