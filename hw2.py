from cProfile import label
import re
import sys

import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    corpus = open(corpus_path).read()
    sentences = corpus.split('\n')
    processed_corpus = []
    for i in range(len(sentences)):
        try:
            sentence, label = sentences[i].split('\t')
            label = int(label)
            words = nltk.word_tokenize(sentence)
            processed_corpus.append((words, label))
        except:
            continue
    return processed_corpus



# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    if word in negation_words:
        return True
    if word[-3: ] == "n't":
        return True
    return False

# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    pos_snippet = nltk.pos_tag(snippet)
    NOT_TAG = "NOT_"
    for i in range(len(pos_snippet)):
        word, pos = pos_snippet[i]
        if word in negation_enders or word in sentence_enders or pos =='JJR' or pos == 'RBR':
            break
        if is_negation(word):
            if i + 1 < len(pos_snippet) and word == "not" and pos_snippet[i][0] == "only":
                break
            elif i + 1 < len(pos_snippet):
                snippet[i + 1] = NOT_TAG + snippet[i + 1]
    return snippet

# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    index = 0
    dictionary = {}
    for i in range(len(corpus)):
        snippet, label = corpus[i]
        for word in snippet:
            if word not in dictionary:
                dictionary[word] = index
                index += 1
    return dictionary
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    word_occurrences = np.zeros(len(feature_dict))
    for word in snippet:
        if word in feature_dict:
            word_occurrences[feature_dict[word]] += 1
    return word_occurrences


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    X = np.empty((len(corpus), len(feature_dict)))
    Y = np.empty(len(corpus))
    for i in range(len(corpus)):
        snippet, label = corpus[i]
        X[i] = vectorize_snippet(snippet, feature_dict)
        Y[i] = label
    return X, Y


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X: np.ndarray):

    for i in range(X.shape[1]):
        X_column = X[:, i]
        minimum = X_column.min()
        maximum = X_column.max()
        if maximum == minimum:
            X[:, i] = np.zeros(X_column.shape[0])
        else:
            X[:, i] = (X_column - minimum)/(maximum - minimum)


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    corpus = load_corpus(corpus_path)
    tagged_corpus = []
    for i in range(len(corpus)):
        snippet, label = corpus[i]
        tagged_corpus.append((tag_negation(snippet), label))
    feature_dict = get_feature_dictionary(tagged_corpus)
    X, Y = vectorize_corpus(tagged_corpus, feature_dict)
    normalize(X)
    model = LogisticRegression()
    model.fit(X, Y)
    return model, feature_dict


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(Y_pred)):
        if Y_test[i] == 1 and Y_pred[i] == Y_test[i]:
            tp += 1
        elif Y_test[i] == 0 and Y_pred[i] != Y_test[i]:
            fp += 1
        elif Y_test[i] == 1 and Y_pred[i] != Y_test[i]:
            fn += 1

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f_measure = (2 * precision * recall)/(precision + recall)
    return precision, recall, f_measure


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model: LogisticRegression, feature_dict, corpus_path):
    corpus = load_corpus(corpus_path)
    tagged_corpus = []
    for i in range(len(corpus)):
        snippet, label = corpus[i]
        tagged_corpus.append((tag_negation(snippet), label))
    X_test, Y_test = vectorize_corpus(tagged_corpus, feature_dict)
    Y_pred = model.predict(X_test)
    return evaluate_predictions(Y_pred, Y_test)


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model: LogisticRegression, feature_dict: dict, k=1):
    weight_array = []
    for index in range(logreg_model.coef_.shape[1]):
        weight_array.append((index, logreg_model.coef_[0, index]))
    weight_array.sort(key = lambda x: x[1], reverse = True)
    top_k_words = []
    reverse_feature_dict = {v: k for k, v in feature_dict.items()}
    for i in range(len(weight_array)):
        index, weight = weight_array[i]
        top_k_words.append((reverse_feature_dict[i], weight))
    return top_k_words[: k]


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
