import numpy as np
import re
import itertools
import random
import constants
from collections import Counter


PY_REC = constants.PY_REC
REC_LENGTH = constants.REC_LENGTH
WIDTH = PY_REC * REC_LENGTH
THRESHOLD = constants.THRESHOLD
random.seed("The seed is the same for all workers so that all samples are used")

def reshape(val, x):
    """
    Make sure features are between 0.0 and 1.0
    Based on max seen values
    """
    for i in range(len(x)/REC_LENGTH):
        x[REC_LENGTH*i] = float(x[REC_LENGTH*i])/4.0
        x[REC_LENGTH*i+1] = float(x[REC_LENGTH*i+1])/400.0
        x[REC_LENGTH*i+2] = float(x[REC_LENGTH*i+2])/360.0
        x[REC_LENGTH*i+3] = float(x[REC_LENGTH*i+3])/360.0
        x[REC_LENGTH*i+4] = float(x[REC_LENGTH*i+4])/(2*3.14159)
        x[REC_LENGTH*i+5] = float(x[REC_LENGTH*i+5])/(2*3.14159)
        x[REC_LENGTH*i+6] = float(x[REC_LENGTH*i+6])/(2*3.14159)
        x[REC_LENGTH*i+7] = float(x[REC_LENGTH*i+7])/(2*3.14159)
        x[REC_LENGTH*i+8] = float(x[REC_LENGTH*i+8])/(2*3.14159)
    x.insert(0, float(val))
    return x


def batch(iterable, n=1):
    """
    Allow to iterate over an iterable by a given step at a time.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
        
def pad_sentences(sentences, padding_val=0):
    """
    Pads all samples to the defined length.
    """
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = (WIDTH+1) - len(sentence)
        if num_padding < 0:
            new_sentence = sentence[:(WIDTH+1)]
            padded_sentences.append(new_sentence)
        elif num_padding > 0:
            new_sentence = sentence + [padding_val] * num_padding
            padded_sentences.append(new_sentence)
        else:
            padded_sentences.append(sentence)
    return padded_sentences


def load_data(files):
    """
    The dataset should be given as CSV values
    Loads data from files and generates labels.
    Returns array inputs and labels.
    """
    # Load data from files
    x_text_pre = []
    for f in files:
       x_text_pre += list(open(f, "r").readlines())
    # Split by value
    x_text = [s.split(",") for s in x_text_pre]
    # Make the features more dependent by aggregating them
    for i in range(len(x_text)):
        x_text_pre[i] = reshape(x_text[i][0], x_text[i][1:])
    # Generate labels
    x = pad_sentences(x_text_pre)
    return random.shuffle(x)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
