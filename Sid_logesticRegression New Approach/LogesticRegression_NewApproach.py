import nltk
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
import string
import pandas as pd
import ssl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
sns.set(style="white")



def find_features(female_wordset, male_wordset, text):
    feature = {}
    feature["mset_count"] = 0
    feature["fset_count"] = 0
    for word in male_wordset:
        if word in text :
            feature["mset_count"] += 1
    for word in female_wordset :
        if word in text :
            feature["fset_count"] += 1
    return feature

def add_to_bag(bag_of_words, tweet, description,stop) :

    # adding the word to the bag except stopwords
    for word in tweet.split():
        if word.isalpha() and word.lower() not in stop:
            bag_of_words.append(word.lower())
    for word in description.split():
        if word.isalpha() and word.lower() not in stop:
            bag_of_words.append(word.lower())

    return bag_of_words


def main() :

    df = pd.read_csv('/Users/sid/Desktop/gender-classifier-DFE-791531.csv', encoding = 'latin1')
    #df = shuffle(shuffle(shuffle(df)))
    print(df.head(10))

    all_descriptions = df['description']
    all_tweets = df['text']
    all_genders = df['gender']
    all_gender_confidence = df['gender:confidence']
    description_tweet_gender = []

    # comment out if running the program for the first time in order to download the stopwords data file
    '''
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download() 
    '''
    # Creation of bag of words for description
    bag_of_words_male = []
    bag_of_words_female = []
    c = 0  # for the index of the row
    stop = stopwords.words('english')
    for tweet in all_tweets:
        description = all_descriptions[c]
        gender = all_genders[c]
        gender_confidence = all_gender_confidence[c]

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

        # remove the rows which has an empty tweet and description
        # remove the rows with unknown or empty gender
        # remove the rows which have gender:confidence < 80%
        if (str(tweet) == 'nan' and str(description) == 'nan') or str(gender) == 'nan' or str(
                gender) == 'unknown' or float(gender_confidence) < 0.8:
            c += 1
            continue

        if str(gender) == 'male':
            bag_of_words_male = add_to_bag(bag_of_words_male, tweet, description,stop)
            description_tweet_gender.append((tweet + " " + description, gender))

        elif str(gender) == 'female':
            bag_of_words_female = add_to_bag(bag_of_words_female, tweet, description,stop)
            description_tweet_gender.append((tweet + " " + description, gender))

        c += 1

    print(len(bag_of_words_male))
    print(len(bag_of_words_female))

    common_words_ratio = []
    common_words = list(set(bag_of_words_male).intersection(bag_of_words_female))
    uniquewords_male = list(set(bag_of_words_male).difference(set(bag_of_words_female)))
    uniquewords_female = list(set(bag_of_words_female).difference(set(bag_of_words_male)))

    print("Number of common words", len(common_words))
    print("Number of unique words", len(uniquewords_male))
    print("Number of unique words", len(uniquewords_female))

    c=0
    for word in common_words :
        r = bag_of_words_male.count(word)/bag_of_words_female.count(word)
        if r > 1:
            common_words_ratio.append(r)
        else :
            common_words_ratio.append(1/r)
        c += 1

    print("Maximum ratio",max(common_words_ratio))
    print("Minimum ratio",min(common_words_ratio))

    N_common_words_ratio = np.array(common_words_ratio)

    mu = np.mean(N_common_words_ratio)  # mean of distribution
    sigma = np.std(N_common_words_ratio)  # standard deviation of distribution
    num_bins = 35

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(N_common_words_ratio,num_bins)

    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    ax.plot(bins, y, '--')
    ax.set_xlabel('ratio_of_common_words_count used by Male & Female')
    ax.set_ylabel('frequency')
    ax.set_title("Histogram: $\mu=$"+ str(mu)+" $\sigma=$"+ str(sigma))

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    #plt.show()


    c=0
    for ratio in common_words_ratio:
        if ratio < 1.3:
            del common_words[c]
        else:
            c+= 1

    male_wordset = list(set(uniquewords_male).union(set(common_words)))
    female_wordset = list(set(uniquewords_female).union(set(common_words)))


    # creating the feature set, training set and the testing set
    feature_set = [(find_features(male_wordset, female_wordset, text), gender) for (text, gender) in description_tweet_gender]
    training_set = feature_set[:int(len(feature_set) * 4 / 5)]
    testing_set = feature_set[int(len(feature_set) * 4 / 5):]

    print("Size of feature set", len(feature_set))
    print("Size of training set", len(training_set))
    print("Size of testing set", len(testing_set))


    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    accuracy = nltk.classify.accuracy(LogisticRegression_classifier, testing_set) * 100
    print("Logistic Regression classifier accuracy =", accuracy)


    X = []
    y = []

    # Creating a different version of the feature set
    c = 0
    for (dicti, gender) in feature_set :
        X.append([])
        a = dicti['mset_count']
        b = dicti['fset_count']
        X[c].append(a)
        X[c].append(b)
        y.append(gender)
        c+= 1

    np_X = np.array(X)

    binary_y = []

    for result in y :
        if result == "male" :
            binary_y.append(1)
        else :
            binary_y.append(0)


    # Using a contour plot and scatter diagram to plot the data points and  the decisssion boundary

    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(np_X, binary_y)

    xx, yy = np.mgrid[0:80:1, 0:80:1]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(np_X[45:, 0], np_X[45:, 1], c=binary_y[45:], s=35,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(0, 60), ylim=(0, 60),
           xlabel="Num_words in male_wordset", ylabel="Num_words in female_wordset")

    ax.set_title("Decision Boundary")

    plt.show()

    # code for predicting whether someone is male or female on the basis of
    '''
    for c in range(50):
        text = all_tweets[c]
        description = all_descriptions[c]
        features = find_features(male_wordset, female_wordset, str(description) + " " + str(text))
        z = LogisticRegression_classifier.classify_many(features)
        c += 1 
    '''


if __name__== "__main__":
  main()