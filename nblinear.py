## toxic comment text: train a comment classifier using nbsvm

# load libraries
import os
import re
import itertools
import argparse
import pickle
from time import time

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

path = "/input/"
output_path = "/output/"
CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
COMMENT = 'comment_text'


# import data
def load_data():
    print("Loading data...\n")
    train = pd.read_csv(path+'train.csv')
    test = pd.read_csv(path+'test.csv')
    return train, test


# text cleaning - remove punctuation and symbols
def clean_text(raw):  return re.sub(r'[^\w\s\']+', '', raw.strip()).lower()


# train a vectorizer for test features
def vectorizer_fit(train):
    # represent data with features
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("Creating the bag of words...\n")
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9,
                                 strip_accents='unicode', use_idf=1,
                                 smooth_idf=1, sublinear_tf=1)
    vectorizer.fit(train[COMMENT])
    return vectorizer


# nb feature
def pr(X, y, y_i):
    p = X[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


# fitting nbsvm
def get_model(X, y, mode='logreg', val_data=None):
    y = y.values
    r = np.log(pr(X,y,1) / pr(X,y,0))
    X_nb = X.multiply(r)
    if mode == 'logreg':
        m = LogisticRegression(C=5, dual=True)
        return m.fit(X_nb, y), r
    if mode == 'gblinear':
        Xval_nb = val_data[0].multiply(r)
        dtrain = xgb.DMatrix(X_nb, y)
        dval = xgb.DMatrix(Xval_nb, val_data[1].values)
        param = {'nthread':4, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'auc',
                 'eta':0.2, 'alpha':0.2, 'lambda':0.2, 'booster':'gblinear'}
        seed = np.random.randint(100)
        param['seed'] = seed
        num_round, early_stopping_rounds = 200, 4
        watchlist = [(dtrain, 'train'), (dval, 'validation')]
        m = xgb.train(param, dtrain, num_round, watchlist,
                      early_stopping_rounds=early_stopping_rounds)
        return m, r


# training and cross-validating
def train_linear(train, test, vectorizer, mode='logreg', kfold=False, nb_folds=5, valid_split=0.1):
    print("Start training...")
    test_x = vectorizer.transform(test[COMMENT])
    cv_scores = []
    ptests_all = np.zeros((len(test), len(CLASSES)))

    for j,c in enumerate(CLASSES):
        print("\nTraining target {}...".format(c))

        if kfold:
            pvals = []
            ptests = []
            #kf = KFold(n_splits=nb_folds, shuffle=True, random_state=0)
            kf = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=0)
            cv_indices = [(tr_id, val_id) for tr_id, val_id in kf.split(train[COMMENT], train[c])]
            for i in range(nb_folds):
                print("Training fold {}...".format(i+1))
                train_x, val_x = train[COMMENT][cv_indices[i][0]], train[COMMENT][cv_indices[i][1]]
                train_y, val_y = train[c][cv_indices[i][0]], train[c][cv_indices[i][1]]
                train_x, val_x = vectorizer.transform(train_x), vectorizer.transform(val_x)
                val_data = (val_x, val_y) if mode == 'gblinear' else None
                m,r = get_model(train_x, train_y, mode, val_data)
                if mode == 'logreg':
                    pvals.append(m.predict_proba(val_x.multiply(r))[:,1])
                    ypvals = [[1 if p>=0.5 else 0 for p in pval] for pval in pvals]
                    ptests.append(m.predict_proba(test_x.multiply(r))[:,1])
                else:
                    pvals.append(m.predict(xgb.DMatrix(val_x.multiply(r))))
                    ypvals = [[1 if p>=0.5 else 0 for p in pval] for pval in pvals]
                    ptests.append(m.predict(xgb.DMatrix(test_x.multiply(r))))
            ptests_all[:,j] = np.mean(ptests, 0)
            # record cv
            cv_scores.append(np.mean([roc_auc_score(train[c][cv_indices[i][1]], pvals[i]) for i in range(nb_folds)]))

        else:
            np.random.seed(0)
            mask = np.random.rand(len(train)) < valid_split
            train_x, train_y = train[COMMENT][~mask], train[c][~mask]
            val_x, val_y = train[COMMENT][mask], train[c][mask]
            train_x, val_x = vectorizer.transform(train_x), vectorizer.transform(val_x)
            val_data = (val_x, val_y) if mode == 'gblinear' else None
            m,r = get_model(train_x, train_y, mode, val_data)
            if mode == 'logreg':
                pval = m.predict_proba(val_x.multiply(r))[:,1]
                ypval = [1 if p>=0.5 else 0 for p in pval]
                ptest = m.predict_proba(test_x.multiply(r))[:,1]
            else:
                pval = m.predict(xgb.DMatrix(val_x.multiply(r)))
                ypval = [1 if p>=0.5 else 0 for p in pval]
                ptest = m.predict(xgb.DMatrix(test_x.multiply(r)))
            ptests_all[:,j] = ptest
            # record cv
            cv_scores.append(roc_auc_score(val_y, pval))

    print("CV scores: ", cv_scores)
    print("Avg. CV scores: ", np.mean(cv_scores))

    return ptests_all


def run(mode='logreg', kfold=False, save_pred=True):
    train, test = load_data()
    print("Cleaning and parsing the texts...")
    t0 = time()
    train[COMMENT] = train[COMMENT].apply(clean_text)
    test[COMMENT] = test[COMMENT].apply(clean_text)
    print("Elapsed time %.2f sec for cleaning data\n" %(time()-t0))
    vectorizer = vectorizer_fit(train)
    preds = train_linear(train, test, vectorizer, mode, kfold)
    if save_pred:
        print("Saving data...\n")
        submit = pd.DataFrame(preds, columns=CLASSES)
        submit['id'] = test['id']
        submit.to_csv(output_path + 'toxic_nblinear_%s.csv'%mode, index=False)
    print("The whole process spends %.2fs\n" %(time()-t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Toxic comments detection')
    parser.add_argument('--mode', default='logreg', type=str)
    parser.add_argument('--kfold', default=False, type=bool)
    parser.add_argument('--save_pred', default=True, type=bool)
    args = vars(parser.parse_args())
    run(**args)


