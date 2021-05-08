# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

def get_num_words_per_sample(sample_texts):
    """ Returns the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

def plot_sample_length_distribution(sample_texts):
    """ Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

def get_num_classes(labels):
    res = set()
    for label in labels:
        res.add(label)
    return len(res)

if __name__ == '__main__':
    labels = [2, 5, 6, 6, 5, 4, 2, 5]
    print 'num classes:', get_num_classes(labels=labels)

    texts = ['',]
    print 'get num words per sample:', get_num_words_per_sample(texts)
