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
from sklearn.model_selection import KFold, StratifiedKFold

SEED = 2019
path = Path('/kaggle/input/jigsaw-toxic-comment-classification-challenge')
model_path = Path('/kaggle/input/jigsaw-toxic-comments')
output_path = Path('/kaggle/working')

parser = argparse.ArgumentParser()
parser.add_argument('--n-splits', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=48)
args = parser.parse_args()

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_dfs():
    train_df = pd.read_csv(path/'train.csv')
    test_df = pd.read_csv(path/'test.csv')
    return train_df, test_df


def train_val_split(train_x):
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    cv_indices = [(tr_idx, val_idx) for tr_idx, val_idx in kf.split(train_x)]
    return cv_indices


def main(args):
    # load data for cls
    train_df, test_df = load_dfs()
    data_lm = load_data(model_path, 'data_lm.pkl', bs=args.batch_size)

    # training preparation
    train_targets = train_df[label_cols].values.astype('int8')
    train_preds = np.zeros(train_targets.shape, dtype='float32') # matrix for the out-of-fold predictions
    test_preds = np.zeros((len(test_df), len(label_cols)), dtype='float32') # matrix for the predictions on the testset
    cv_indices = train_val_split(train_df)

    # start training
    print()
    for i, (trn_idx, val_idx) in enumerate(cv_indices):
        print(f'Fold {i + 1}')

        # prepare databunch
        trn_df, val_df = train_df.iloc[trn_idx], train_df.iloc[val_idx]
        data_clas = TextClasDataBunch.from_df(path, train_df=trn_df, valid_df=val_df, test_df=test_df,
                                              vocab=data_lm.vocab, bs=args.batch_size,
                                              text_cols='comment_text', label_cols=label_cols)

        # prepare model
        learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, model_dir=output_path)
        learn.metrics = [partial(accuracy_thresh, thresh=0.2)]
        learn.model_dir = model_path
        learn.load_encoder('fine_tuned_enc')
        learn.model_dir = output_path

        # train
        learn.fit_one_cycle(3, 2e-2, moms=(0.85,0.75))
        learn.freeze_to(-2)
        learn.fit_one_cycle(2, slice(1e-2/(2.6**2),1e-2), moms=(0.85,0.75))
        learn.freeze_to(-3)
        learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.85,0.75))

        # inference
        pred_val, _ = learn.get_preds(ds_type=DatasetType.Valid)
        train_preds[val_idx] = pred_val.numpy()

        data_clas.add_test(data_clas.test_ds)  # correct the order for kaggle submission
        pred_test, _ = learn.get_preds(ds_type=DatasetType.Test)
        test_preds += pred_test.numpy() / args.n_splits
        print()

    # make submission
    print(f'val auc cv score is {roc_auc_score(train_targets, train_preds)}')
    submit = pd.DataFrame(test_preds, columns=label_cols)
    submit['id'] = test_df['id'].tolist()
    submit.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main(args)