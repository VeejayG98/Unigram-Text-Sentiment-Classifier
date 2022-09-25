from cProfile import label
import re
import sys

import nltk
import numpy
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
            # words = sentence.split(' ')
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


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    pos_snippet = nltk.pos_tag(snippet)
    NOT_TAG = "NOT_"
    print(pos_snippet)
    for i in range(len(pos_snippet)):
        word, pos = pos_snippet[i]
        if word in negation_enders or word in sentence_enders or pos =='JJR' or pos == 'RBR':
            break
        if is_negation(word):
            if i + 1 < len(pos_snippet) and word == "not" and pos_snippet[i][0] == "only":
                break
            else:
                snippet[i + 1] = NOT_TAG + snippet[i + 1]
    return snippet

# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    pass
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    pass


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    pass


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    pass


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    pass


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    pass


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    pass


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    pass


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

# corpus = load_corpus('/Users/jayasuryaagovindraj/Documents/NLP Assignments/Assignment 2/Programming/test.txt')
# for i in range(len(corpus)):
#     snippet, label = corpus[i]
#     tag_negation(snippet)