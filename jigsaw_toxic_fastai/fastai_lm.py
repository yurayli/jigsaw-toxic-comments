import os, time, json, re
import itertools, argparse, pickle, random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fastai
from fastai.text import *
from fastai.callbacks import *

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 2019
path = Path('/kaggle/input')
output_path = Path('/kaggle/working')

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-size', type=int, default=60000)
parser.add_argument('--batch-size', type=int, default=48)
parser.add_argument('--epochs', type=int, default=4,
                    help='number of training epochs')
args = parser.parse_args()

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_lm_data():
    train_df = pd.read_csv(path/'train.csv')
    test_df = pd.read_csv(path/'test.csv')
    train_df.drop(label_cols, 1, inplace=True)
    all_df = pd.concat([train_df, test_df])
    all_df.index = np.arange(len(all_df))

    np.random.seed(SEED)
    msk = np.random.rand(len(all_df)) > 0.1
    trn_df = all_df[msk]
    val_df = all_df[~msk]

    return trn_df, val_df


def main(args):
    # load data for lm
    trn_df, val_df = load_lm_data()

    # databunch preparation
    data_lm = TextLMDataBunch.from_df(path, trn_df, val_df, text_cols=1,
                                      max_vocab=args.vocab_size, bs=args.batch_size)
    data_lm.path = output_path
    data_lm.save('data_lm.pkl')

    # start training
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3, model_dir=output_path)
    learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
    learn.unfreeze()
    learn.fit_one_cycle(args.epochs, 1e-3, moms=(0.8,0.7),
                        callbacks=[SaveModelCallback(learn, name='fine_tuned')])
    learn.save_encoder('fine_tuned_enc')

    # test model
    TRIGGER_TEXT = "Wikipedia is a multilingual online encyclopedia"
    N_WORDS = 40
    N_SENTENCES = 2
    print("\n".join(learn.predict(TRIGGER_TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


if __name__ == '__main__':
    main(args)