## load libraries
import os, re
import itertools, argparse, pickle
from time import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, concatenate, SpatialDropout1D
from keras.layers import Embedding, Bidirectional, GRU, LSTM, CuDNNLSTM, CuDNNGRU, Conv1D
from keras.layers import MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


# Loss and evaluation
def log_loss(y_true, y_pred):
    y_pred = np.maximum(np.minimum(y_pred, 1 - 1e-6), 1e-6)
    return -np.mean([np.log(p) if y_true[i]==1 else np.log(1-p) for i,p in enumerate(y_pred)])

class RocAucEvaluation(Callback):
    def __init__(self, validation_data, interval=1):
        super(Callback, self).__init__()
        self.best_score = 0
        self.best_epoch = -1
        self.interval = interval
        if validation_data:
            self.x_val, self.y_val = validation_data
        else:
            raise ValueError('validation_data must be a tuple (x_val, y_val)')
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            if score > self.best_score:
                self.best_score = score
                self.best_epoch = epoch
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

def precision_recall_f1(y, p):
    eps = 1e-7
    tp = ((y==p) & (y==1)).sum()
    fn = ((y!=p) & (y==1)).sum()
    fp = ((y!=p) & (y==0)).sum()
    p = tp / float(tp+fp+eps)
    r = tp / float(tp+fn+eps)
    f = 2*p*r / (p+r+eps)
    return p, r, f


# Learning rate scheduler
def schedule(epoch, base_lr, decay=0.5, staircase=True, steps=10):
    global which_step
    which_step = 0
    if staircase:
        if ((epoch+1)%steps == 0):
            print("decay learning rate by {}.".format(decay))
            which_step += 1
        return base_lr * decay**which_step
    else:
        return base_lr * decay**epoch


# Load pre-trained word vector (GloVe)
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def get_embedding(embedding_file, embedding_dim, tokenizer, vocab_size):
    embeddings_index = dict(get_coefs(*o.strip().split(' ')) for o in open(embedding_file))
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    word_index = tokenizer.word_index
    nb_words = min(vocab_size, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_dim))
    for word, i in word_index.items():
        if i >= vocab_size: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


# Pre-processing
def remove_long_tail(word):
    # remove long tail words, e.g. 'kickkkkkkk' -> 'kick'
    if len(word) < 4:
        return word
    count, position = 1, 2
    tail_letter = list(word)[-1]
    while list(word)[-position] == tail_letter:
        count, position = count + 1, position + 1
        if position == len(word):
            # 'zzzzzzzzz' -> 'zzz'
            return word[:3]
    if count > 2:
        return word[:-count+1]
    else:
        return word

def clean_text(raw):  return re.sub(r'[^\w\s\']+', '', raw.strip()).lower()

def clean_text_(raw):
    words = raw.strip().split()
    # clean number in word, e.g. toxic words "10fags fags fcockc dick0"
    words = [re.sub(r"[\d]", '', w) if re.match(r'\D*\d+\D+', w) or re.match(r'\D+\d+\D*', w) \
        else w for w in words]
    # split lowercase and uppcase, e.g. 'KeepGoing' -> 'Keep Going'
    words = [''.join([' '+c if c.isupper() else c for c in list(w)]) \
        if re.search(r'[a-z]+[A-Z]+', w) else w for w in words]
    # remove long tail words, e.g. 'kickkkkkkk' -> 'kick'
    words = [remove_long_tail(w) for w in words]
    return ' '.join(words)

def tokenize(train, test, vocab_size):
    # Clean and tokenize the comment texts as input
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train)
    train_tokens = tokenizer.texts_to_sequences(train)
    test_tokens = tokenizer.texts_to_sequences(test)
    return train_tokens, test_tokens, tokenizer


# Visualization plot
def plot_history(history, fname):
    # plot history of loss and acc
    accs = history.history['acc']
    val_accs = history.history['val_acc']
    losses = history.history['loss']
    val_losses = history.history['val_loss']
    epochs = range(len(accs))
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs, accs)
    ax1.plot(epochs, val_accs)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend(['train', 'val'], loc='best')
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(epochs, losses)
    ax2.plot(epochs, val_losses)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend(['train', 'val'], loc='best')
    fig.savefig(fname)

def plot_confusion_matrix(cm, classes, fname, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.style.use('ggplot')
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fname)
