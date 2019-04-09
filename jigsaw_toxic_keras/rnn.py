## toxic comment texts:
## train a classifier using recurrent neural network
## with pre-trained GloVe vec

from utils import *

path = "/train_files/"
output_path = "/output/"
EMBEDDING_FILE = "/glove/glove.6B.100d.txt"
EMBEDDING_DIM = 100
CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
COMMENT = 'comment_text'
base_lr = 5e-4


def load_data():
    print("Loading data...\n")
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    return train, test


def build_model(vocab_size, seq_len, embedding_matrix):
    inp = Input(shape=(seq_len,), dtype='int32', name='model_input')
    emb = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
    emb = SpatialDropout1D(0.25)(emb)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(emb)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)
    net = Model(inp, x)
    net.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
    return net


def compute_cv_auc(ys, ps, cv_indices=None, nb_folds=None, kfold=False):
    if kfold:
        return [np.mean([roc_auc_score(ys[:,j][cv_indices[i][1]], ps[i][:,j])
                        for i in range(nb_folds)]) for j in range(len(CLASSES))]
    else:
        return [roc_auc_score(ys[:,j], ps[:,j]) for j in range(len(CLASSES))]


earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
checkpoint = ModelCheckpoint(output_path + 'weights_rnn.h5',
                             verbose=1,
                             save_weights_only=True)
schedule.__defaults__ = (base_lr, 0.5, True, 4)
scheduler = LearningRateScheduler(schedule)
optim = Adam(lr=base_lr)


def train_rnn(train_tokens, train_targets, test_tokens, embedding_matrix, vocab_size, valid_split,
              nb_epoch, kfold=False, nb_folds=6, seq_len=150):

    train_tensor = sequence.pad_sequences(train_tokens, maxlen=seq_len, value=0)
    x_test = sequence.pad_sequences(test_tokens, maxlen=seq_len, value=0)
    target_values = train_targets.values

    print("\nStart training...")
    # mix k-fold models
    if kfold:
        pvals = []
        ypvals = []
        ptests = []
        kf = KFold(n_splits=nb_folds, shuffle=True, random_state=13)
        cv_indices = [(tr_id, val_id) for tr_id, val_id in kf.split(train_tensor)]

        for i in range(nb_folds):
            x_train, x_val = train_tensor[cv_indices[i][0]], train_tensor[cv_indices[i][1]]
            y_train, y_val = target_values[cv_indices[i][0]], target_values[cv_indices[i][1]]

            auc_record = RocAucEvaluation((x_val, y_val))
            rnn = build_model(vocab_size, seq_len, embedding_matrix)
            rnn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=nb_epoch,
                     batch_size=128, callbacks=[auc_record, checkpoint, earlystopping])
            rnn.load_weights(output_path + 'weights_rnn.h5'.format(auc_record.best_epoch))

            pvals.append(rnn.predict(x_val))
            ptests.append(rnn.predict(x_test))

        cv_auc = compute_cv_auc(target_values, pvals, cv_indices, nb_folds, kfold)
        preds = np.mean(np.stack(ptests), 0)

    else:
        np.random.seed(13)
        mask = np.random.rand(len(train_tensor)) < valid_split
        x_train, y_train = train_tensor[~mask], target_values[~mask]
        x_val, y_val = train_tensor[mask], target_values[mask]

        auc_record = RocAucEvaluation((x_val, y_val))
        rnn = build_model(vocab_size, seq_len, embedding_matrix)
        rnn.summary()
        history = rnn.fit(x_train, y_train, validation_data=(x_val, y_val),
                          epochs=nb_epoch, batch_size=128,
                          callbacks=[auc_record, checkpoint, earlystopping])
        rnn.load_weights(output_path + 'weights_rnn.h5'.format(auc_record.best_epoch))

        pval = rnn.predict(x_val)
        preds = rnn.predict(x_test, verbose=1)
        cv_auc = compute_cv_auc(y_val, pval)

        # plot history of fc
        plot_history(history, output_path+'toxic_rnn_curve.png')

    print("CV scores: ", cv_auc)
    print("Avg. CV scores: ", np.mean(cv_auc))

    return preds


def run(vocab_size, valid_split, nb_epoch, kfold, save_pred):
    train, test = load_data()
    train_targets = train[CLASSES]
    t0 = time()
    train_tokens, test_tokens, tokenizer = tokenize(train[COMMENT], test[COMMENT], vocab_size)
    print("Elapsed time {:.3f} seconds for tokenizing.".format(time()-t0))
    embedding_matrix = get_embedding(EMBEDDING_FILE, EMBEDDING_DIM, tokenizer, vocab_size)
    preds = train_rnn(train_tokens, train_targets, test_tokens, embedding_matrix,
                      vocab_size, valid_split, nb_epoch, kfold)
    if save_pred:
        print("Saving submission...")
        submit = pd.DataFrame(preds, columns=train_targets.columns)
        submit['id'] = pd.read_csv(path + 'test.csv')['id']
        submit.to_csv(output_path + 'toxic_rnn.csv', index=False)
    print("Completed!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Toxic comments detection')
    parser.add_argument('--vocab_size', default=50000, type=int)
    parser.add_argument('--valid_split', default=0.1, type=float)
    parser.add_argument('--nb_epoch', default=7, type=int)
    parser.add_argument('--kfold', default=False, type=bool)
    parser.add_argument('--save_pred', default=False, type=bool)
    args = vars(parser.parse_args())
    run(**args)


