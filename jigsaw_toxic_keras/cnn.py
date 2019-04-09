## toxic comments detection:
## train a classifier using convolutional neural network
## with pre-trained GloVe vec

from utils import *

path = "/train_files/"
output_path = "/output/"
EMBEDDING_FILE = "/glove/glove.42B.300d.txt"
EMBEDDING_DIM = 300
CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
COMMENT = 'comment_text'
base_lr = 2e-4


def load_data():
    print("Loading data...\n")
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    return train, test


def build_conv_1(vocab_size, seq_len, embedding_matrix):
    inp = Input(shape=(seq_len,), dtype='int32')
    emb = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=seq_len)(inp)
    emb = SpatialDropout1D(0.5)(emb)
    x = Dropout(0.2)(emb)
    x = Conv1D(96, 5, padding='same', activation='relu')(x)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(0.1)(x)
    outp = Dense(6, activation="sigmoid")(x)
    conv = Model(inputs=inp, outputs=outp)
    conv.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    return conv


def build_conv_2(vocab_size, seq_len, embedding_matrix):
    inp = Input(shape=(seq_len,), dtype='int32', name='model_input')
    emb = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=seq_len)(inp)
    emb = SpatialDropout1D(0.25)(emb)
    x = Dropout(0.2)(emb)

    branch1 = Conv1D(64, 1, padding='same', activation='relu')(x)
    branch2 = Conv1D(48, 2, padding='same', activation='relu')(x)
    branch3 = Conv1D(48, 3, padding='same', activation='relu')(x)
    branch5 = Conv1D(32, 5, padding='same', activation='relu')(x)
    x = concatenate([branch1, branch2, branch3, branch5], axis=2)

    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)
    conv = Model(inp, x)
    conv.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    return conv


def build_model(vocab_size, seq_len, embedding_matrix, mode=1):
    if mode==1:
        return build_conv_1(vocab_size, seq_len, embedding_matrix)
    if mode==2:
        return build_conv_2(vocab_size, seq_len, embedding_matrix)


def compute_cv_auc(ys, ps, cv_indices=None, nb_folds=None, kfold=False):
    if kfold:
        return [np.mean([roc_auc_score(ys[:,j][cv_indices[i][1]], ps[i][:,j])
                        for i in range(nb_folds)]) for j in range(len(CLASSES))]
    else:
        return [roc_auc_score(ys[:,j], ps[:,j]) for j in range(len(CLASSES))]


earlystopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
checkpoint = ModelCheckpoint(output_path + 'weights_cnn.h5',
                             verbose=1, save_best_only=True,
                             save_weights_only=True)
schedule.__defaults__ = (base_lr, 0.5, True, 5)
scheduler = LearningRateScheduler(schedule)
optim = Adam(lr=base_lr)


def train_cnn(train_tokens, train_targets, test_tokens, embedding_matrix, vocab_size, valid_split,
              model_mode, nb_epoch, kfold=False, nb_folds=6, seq_len=150):

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
            conv = build_model(vocab_size, seq_len, embedding_matrix, model_mode)
            conv.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=nb_epoch,
                     batch_size=128, callbacks=[auc_record, checkpoint, earlystopping])
            conv.load_weights(output_path + 'weights_cnn.h5')

            pvals.append(conv.predict(x_val))
            ptests.append(conv.predict(x_test))

        cv_auc = compute_cv_auc(target_values, pvals, cv_indices, nb_folds, kfold)
        preds = np.mean(np.stack(ptests), 0)

    # use one fold to find the optimal hyper-parameters
    else:
        np.random.seed(13)
        mask = np.random.rand(len(train_tensor)) < valid_split
        x_train, y_train = train_tensor[~mask], target_values[~mask]
        x_val, y_val = train_tensor[mask], target_values[mask]

        auc_record = RocAucEvaluation((x_val, y_val))
        conv = build_model(vocab_size, seq_len, embedding_matrix, model_mode)
        conv.summary()
        history = conv.fit(x_train, y_train, validation_data=(x_val, y_val),
                           epochs=nb_epoch, batch_size=128,
                           callbacks=[auc_record, checkpoint, earlystopping, scheduler])
        conv.load_weights(output_path + 'weights_cnn.h5')

        pval = conv.predict(x_val)
        preds = conv.predict(x_test, verbose=1)
        cv_auc = compute_cv_auc(y_val, pval)

        # plot history of fc
        plot_history(history, output_path+'toxic_cnn_curve.png')

    print("CV scores: ", cv_auc)
    print("Avg. CV scores: ", np.mean(cv_auc))

    return preds


def run(vocab_size, valid_split, model_mode, nb_epoch, kfold, save_pred):
    train, test = load_data()
    train_targets = train[CLASSES]
    t0 = time()
    train_tokens, test_tokens, tokenizer = tokenize(train[COMMENT], test[COMMENT], vocab_size)
    print("Elapsed time {:.3f} seconds for tokenizing.".format(time()-t0))
    embedding_matrix = get_embedding(EMBEDDING_FILE, EMBEDDING_DIM, tokenizer, vocab_size)
    preds = train_cnn(train_tokens, train_targets, test_tokens, embedding_matrix,
                      vocab_size, valid_split, model_mode, nb_epoch, kfold)
    if save_pred:
        print("Saving data...")
        submit = pd.DataFrame(preds, columns=train_targets.columns)
        submit['id'] = pd.read_csv(path + 'test.csv')['id']
        submit.to_csv(output_path + 'toxic_cnn_m%d.csv'%model_mode, index=False)
    print("Completed!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Toxic comments detection')
    parser.add_argument('--vocab_size', default=50000, type=int)
    parser.add_argument('--valid_split', default=0.1, type=float)
    parser.add_argument('--model_mode', default=1, type=int)
    parser.add_argument('--nb_epoch', default=20, type=int)
    parser.add_argument('--kfold', default=True, type=bool)
    parser.add_argument('--save_pred', default=False, type=bool)
    args = vars(parser.parse_args())
    run(**args)


