from data_utils import *
from solver import *
from model import *

path = '/input/'
model_path = '/model/'
output_path = '/output/'

SEED = 135
EMBEDDING_FILE_GV = '/embeddings_1/glove.840B.300d.txt'
EMBEDDING_FILE_PR = '/embeddings_2/paragram_300_sl999/paragram_300_sl999.txt'
EMBEDDING_FILE_FT = '/embeddings_2/wiki-news-300d-1M/wiki-news-300d-1M.vec'

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=180,
                    help='maximum length of a question sentence')
parser.add_argument('--vocab-size', type=int, default=140000)
parser.add_argument('--n-splits', type=int, default=5,
                    help='splits of n-fold cross validation')
parser.add_argument('--batch-size', type=int, default=256,
                    help='batch size during training')
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs')
args = parser.parse_args()

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_and_preproc():
    train_df = pd.read_csv(path+'train.csv')
    test_df = pd.read_csv(path+'test.csv')

    print('cleaning text...')
    t0 = time.time()
    train_df['comment_text'] = train_df['comment_text'].apply(clean_text)
    test_df['comment_text'] = test_df['comment_text'].apply(clean_text)
    print('cleaning complete in {:.0f} seconds.'.format(time.time()-t0))

    return train_df, test_df


def tokenize_comments(train_df, test_df):
    y_train = train_df[label_cols].values.astype('int8')
    full_text = train_df['comment_text'].tolist() + test_df['comment_text'].tolist()

    print('tokenizing...')
    t0 = time.time()
    idx_to_word, word_to_idx = word_idx_map(full_text, args.vocab_size)
    x_train = tokenize(train_df['comment_text'], word_to_idx, args.maxlen)
    x_test = tokenize(test_df['comment_text'], word_to_idx, args.maxlen)
    print('tokenizing complete in {:.0f} seconds.'.format(time.time()-t0))

    return x_train, y_train, x_test, word_to_idx


def train_val_split(train_x):
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    cv_indices = [(tr_idx, val_idx) for tr_idx, val_idx in kf.split(train_x)]
    return cv_indices


def main(args):
    # load data
    train_df, test_df = load_and_preproc()
    train_seq, train_tars, x_test, word_to_idx = tokenize_comments(train_df, test_df)

    # load pretrained embedding
    print('loading embeddings...')
    t0 = time.time()
    embed_mat_1 = get_embedding(EMBEDDING_FILE_GV, 300, word_to_idx, args.vocab_size)
    embed_mat_2 = get_embedding(EMBEDDING_FILE_PR, 300, word_to_idx, args.vocab_size)
    #embed_mat_3 = get_embedding(EMBEDDING_FILE_FT, 300, word_to_idx, args.vocab_size)
    embed_mat = np.mean([embed_mat_1, embed_mat_2], 0)
    print('loading complete in {:.0f} seconds.'.format(time.time()-t0))

    # training preparation
    train_preds = np.zeros(train_tars.shape, dtype='float32') # matrix for the out-of-fold predictions
    test_preds = np.zeros((len(test_df), len(label_cols)), dtype='float32') # matrix for the predictions on the testset
    test_loader = prepare_loader(x_test, train=False)
    cv_indices = train_val_split(train_seq)

    print()
    for i, (trn_idx, val_idx) in enumerate(cv_indices):
        print(f'Fold {i + 1}')

        # train/val split
        x_train, x_val = train_seq[trn_idx], train_seq[val_idx]
        y_train, y_val = train_tars[trn_idx], train_tars[val_idx]
        train_loader = prepare_loader(x_train, y_train, args.batch_size)
        val_loader = prepare_loader(x_val, y_val, train=False)

        # model setup
        model = CommentNet(300, 160, args.vocab_size, embed_mat)
        for name, param in model.named_parameters():
           if 'emb' in name:
               param.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        scheduler = OneCycleScheduler(optimizer, args.epochs, train_loader, max_lr=args.lr, moms=(.8, .7))
        solver = NetSolver(model, optimizer, scheduler)

        # train
        t0 = time.time()
        solver.train_one_cycle(loaders=(train_loader, val_loader), epochs=args.epochs)
        time_elapsed = time.time() - t0
        print('time = {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

        # finetuning
        # solver.model.emb.weight.requires_grad = True
        # solver.optimizer.add_param_group({'params': solver.model.emb.parameters()})
        # ft_lrs = [args.lr/10, args.lr/10/(2.6)**2]
        # scheduler = OneCycleScheduler(solver.optimizer, 2, train_loader, max_lr=ft_lrs, moms=(.8, .7))
        # solver.scheduler = scheduler

        # t0 = time.time()
        # solver.train_one_cycle(loaders=(train_loader, val_loader), epochs=2)
        # time_elapsed = time.time() - t0
        # print('time = {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

        # inference
        solver.model.eval()
        test_scores = []
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device=device, dtype=torch.long)
                score = torch.sigmoid(solver.model(x))
                test_scores.append(score.cpu().numpy())
        test_scores = np.concatenate(test_scores)

        train_preds[val_idx] = solver.val_scores
        test_preds += test_scores / args.n_splits

        print()

    # submit
    print(f'For whole train set, val auc score is {roc_auc_score(train_tars, train_preds)}')
    submit = pd.DataFrame(test_preds, columns=label_cols)
    submit['id'] = test_df['id'].copy()
    submit.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main(args)

