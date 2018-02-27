import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
import pandas as pd
import json

def clean_str(string):
    """
    cleans the input string; expands the short forms
    :arg:  input text string
    :return:  text string
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
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


def load_data_labels(filename, flag):
    df = pd.read_csv(filename, compression='zip', dtype={'Questions': object})
    selected = ['category', 'Questions']
    df = df.reindex(np.random.permutation(df.index))
    x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y_raw = df[selected[0]].tolist()
    if flag == 1:	    
        target = list(set(y_raw))
        print (target)
        labels = []
    else:
        target = json.loads(open('./labels.json').read())
        labels = []
    for i in range(len(x_raw)):
        label = [0 for j in range(len(target))]
        label[y_raw[i]-1] = 1
        labels.append(label)    
    y = np.array(labels)
    return [x_raw,y,target]
    
def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split(":")
        word = values[0]
        vector = np.asarray(values[1:])
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors
