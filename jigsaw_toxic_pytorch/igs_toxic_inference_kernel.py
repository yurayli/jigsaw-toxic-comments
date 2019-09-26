# For inference
import os, time, json, re, copy
import itertools, argparse, pickle, random

import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Sampler

##
USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

##
SEED = 2019
path = '../input/jigsaw-toxic-comment-classification-challenge/'
model_path = '../input/toxic-op-1/'
output_path = './'
EMBEDDING_FILES = [
    '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl',
    # '../input/pickled-paragram-300-vectors-sl999/paragram_300_sl999.pkl',
    '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
]

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=220)
parser.add_argument('--vocab-size', type=int, default=100000)
parser.add_argument('--nb-models', type=int, default=5,
                    help='number of models (folds) to ensemble')
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--enable-ckpt-ensemble', type=bool, default=0)
parser.add_argument('--ckpt-per-fold', type=bool, default=1)
parser.add_argument('--wavg-per-fold', type=bool, default=0)
args = parser.parse_args()

##
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
symbols_to_isolate = ''.join(puncts)
isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}


# Text preprocessing
def clean_punc(x):
    x = str(x)
    x = x.translate(isolate_dict)
    return x

def clean_text(raw):
    x = clean_punc(raw.lower())
    return x.strip()


# Tokenize texts
def word_idx_map(raw_comments, vocab_size):
    def build_vocab(sentences):
        """
        :param sentences: list of list of words
        :return: dictionary of words and their count
        """
        vocab = {}
        for sentence in sentences:
            for word in sentence:
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1
        return vocab

    def most_common_vocab(vocab, k):
        """
        :param vocab: dictionary of words and their count
        :k: former k words to return
        :return: list of k most common words
        """
        sorted_vocab = sorted([(cnt,w) for w,cnt in vocab.items()])[::-1]
        return [(w,cnt) for cnt,w in sorted_vocab][:k]

    texts = [c.split() for c in raw_comments]
    word_freq = build_vocab(texts)
    vocab_freq = most_common_vocab(word_freq, vocab_size)
    idx_to_word = ['<pad>'] + [word for word, cnt in vocab_freq] + ['<unk>']
    word_to_idx = {word:idx for idx, word in enumerate(idx_to_word)}

    return word_to_idx, idx_to_word


def tokenize(comments, word_to_idx, maxlen):
    '''
    Tokenize and numerize the comment sequences
    Inputs:
    - comments: pandas series with wiki comments
    - word_to_idx: mapping from word to index
    - maxlen: max length of each sequence of tokens

    Returns:
    - tokens: array of shape (data_size, maxlen)
    '''

    def text_to_id(c, word_to_idx, maxlen):
        return [(lambda x: word_to_idx[x] if x in word_to_idx else word_to_idx['<unk>'])(w) \
                 for w in c.split()[-maxlen:]]

    return [text_to_id(c, word_to_idx, maxlen) for c in comments]


# Load pre-trained word vector
def load_embeddings(path):
    with open(path,'rb') as f:
        emb_index = pickle.load(f)
    return emb_index

def get_embedding(embedding_file, word_to_idx, embedding_dim=300):
    print(f'loading {embedding_file}')
    embeddings_index = load_embeddings(embedding_file)

    all_embs = np.stack([emb for emb in embeddings_index.values() if len(emb)==embedding_dim])
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    nb_words = len(word_to_idx)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_dim))
    for word, i in word_to_idx.items():
        if i > nb_words: break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        embedding_vector = embeddings_index.get(word.upper())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        embedding_vector = embeddings_index.get(word.title())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
    return embedding_matrix


# Prepare dataset and dataloader
class Toxic_comments(Dataset):

    def __init__(self, tokenized_comments, targets=None, split=None, maxlen=256):
        self.comments = tokenized_comments
        self.targets = targets
        self.split = split
        assert self.split in {'train', 'valid', 'test'}
        self.maxlen = maxlen

    def __getitem__(self, index):
        comment = self.comments[index]
        if self.targets is not None:
            target = self.targets[index]
            return comment, torch.FloatTensor(target)
        else:
            return comment

    def __len__(self):
        return len(self.comments)

    def get_lens(self):
        lengths = np.fromiter(
            ((min(self.maxlen, len(seq))) for seq in self.comments),
            dtype=np.int32)
        return lengths

    def collate_fn(self, batch):
        """
        Collate function for sequence bucketing
        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of comments, and targets
        """

        if self.split in ('train', 'valid'):
            comments, targets = zip(*batch)
        else:
            comments = batch

        lengths = [len(c) for c in comments]
        maxlen = max(lengths)
        padded_comments = []
        for i, c in enumerate(comments):
            padded_comments.append([0]*(maxlen - lengths[i])+c)

        if self.split in ('train', 'valid'):
            return torch.LongTensor(padded_comments), torch.stack(targets)
        else:
            return torch.LongTensor(padded_comments)


class BucketSampler(Sampler):

    def __init__(self, data_source, sort_lens, bucket_size=None, batch_size=1024, shuffle_data=True):
        super().__init__(data_source)
        self.shuffle = shuffle_data
        self.batch_size = batch_size
        self.sort_lens = sort_lens
        self.bucket_size = bucket_size if bucket_size is not None else len(sort_lens)
        self.weights = None

        if not shuffle_data:
            self.index = self.prepare_buckets()
        else:
            self.index = None

    def set_weights(self, weights):
        assert weights >= 0
        total = np.sum(weights)
        if total != 1:
            weights = weights / total
        self.weights = weights

    def __iter__(self):
        indices = None
        if self.weights is not None:
            total = len(self.sort_lens)
            indices = np.random.choice(total, (total,), p=self.weights)
        if self.shuffle:
            self.index = self.prepare_buckets(indices)
        return iter(self.index)

    def get_reverse_indexes(self):
        indexes = np.zeros((len(self.index),), dtype=np.int32)
        for i, j in enumerate(self.index):
            indexes[j] = i
        return indexes

    def __len__(self):
        return len(self.sort_lens)

    def prepare_buckets(self, indices=None):
        lengths = - self.sort_lens
        assert self.bucket_size % self.batch_size == 0 or self.bucket_size == len(lengths)

        if indices is None:
            if self.shuffle:
                indices = shuffle(np.arange(len(lengths), dtype=np.int32))
                lengths = lengths[indices]
            else:
                indices = np.arange(len(lengths), dtype=np.int32)

        #  bucket iterator
        def divide_chunks(l, n):
            if n == len(l):
                yield np.arange(len(l), dtype=np.int32), l
            else:
                # looping till length l
                for i in range(0, len(l), n):
                    data = l[i:i + n]
                    yield np.arange(i, i + len(data), dtype=np.int32), data

        new_indices = []
        extra_batch_idx = None
        for chunk_index, chunk in divide_chunks(lengths, self.bucket_size):
            # sort indices in bucket by descending order of length
            indices_sorted = chunk_index[np.argsort(chunk)]

            batch_idxes = []
            for _, batch_idx in divide_chunks(indices_sorted, self.batch_size):
                if len(batch_idx) == self.batch_size:
                    batch_idxes.append(batch_idx.tolist())
                else:
                    assert extra_batch_idx is None
                    assert batch_idx is not None
                    extra_batch_idx = batch_idx.tolist()

            # shuffling batches within buckets
            if self.shuffle:
                batch_idxes = shuffle(batch_idxes)
            for batch_idx in batch_idxes:
                new_indices.extend(batch_idx)

        if extra_batch_idx is not None:
            new_indices.extend(extra_batch_idx)

        if not self.shuffle:
            self.original_indices = np.argsort(indices_sorted).tolist()
        return indices[new_indices]


def prepare_loader(x, y=None, batch_size=1024, split=None):
    assert split in {'train', 'valid', 'test'}
    dataset = Toxic_comments(x, y, split, args.maxlen)
    if split == 'train':
        sampler = BucketSampler(dataset, dataset.get_lens(),
                                bucket_size=batch_size*30, batch_size=batch_size)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          collate_fn=dataset.collate_fn)
    else:
        sampler = BucketSampler(dataset, dataset.get_lens(),
                                batch_size=batch_size, shuffle_data=False)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          collate_fn=dataset.collate_fn), sampler.original_indices


# Model
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(-1)    # (N, T, M, 1)
        x = x.transpose(1,2)   # (N, M, T, 1)
        x = super(SpatialDropout, self).forward(x)  # (N, M, T, 1), some features are masked
        x = x.squeeze(-1)     # (N, M, T)
        x = x.transpose(1,2)   # (N, T, M)
        return x

class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, embed_dim, embed_matrix):
        super(EmbeddingLayer, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.emb.weight = nn.Parameter(torch.tensor(embed_matrix, dtype=torch.float32))
        self.emb_dropout = SpatialDropout(0.35)

    def forward(self, seq):
        emb = self.emb(seq)
        emb = self.emb_dropout(emb)
        return emb

class RecurrentNet(nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super(RecurrentNet, self).__init__()
        # Init layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_dim*2, hidden_dim, bidirectional=True, batch_first=True)

        for mod in (self.lstm, self.gru):
            for name, param in mod.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)

    def forward(self, seq):
        o_lstm, _ = self.lstm(seq)
        o_gru, _ = self.gru(o_lstm)
        return o_gru

class CommentClassifier(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(CommentClassifier, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(hidden_dim*4, output_dim)

    def forward(self, seq):
        avg_pool = torch.mean(seq, 1)
        max_pool, _ = torch.max(seq, 1)
        h_concat = torch.cat((avg_pool, max_pool), 1)
        out = self.fc_out(self.dropout(h_concat))
        return out

class JigsawNet(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, embed_matrix):
        super(JigsawNet, self).__init__()
        # Init layers
        self.emb_layer = EmbeddingLayer(vocab_size, embed_dim, embed_matrix)
        self.rnns = RecurrentNet(embed_dim, hidden_dim)
        self.classifier = CommentClassifier(hidden_dim, 6)

    def forward(self, seq):
        emb = self.emb_layer(seq)
        o_rnn = self.rnns(emb)
        out = self.classifier(o_rnn)

        return out

class NNAverage(object):
    def __init__(self, model, mu=0.5):
        self.mu = mu
        self.weight_copy = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.weight_copy[name] = 0

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.weight_copy[name] += self.mu * param.data

    def set_weights(self, avg_model):
        for name, param in avg_model.named_parameters():
            if param.requires_grad:
                param.data = self.weight_copy[name]

def eval_model(model, data_loader, mode='test'):
    assert mode in ('val', 'test')
    model.eval()
    test_scores = []
    with torch.no_grad():
        for x in data_loader:
            if mode=='val': x = x[0]
            x = x.to(device=device, dtype=torch.long)
            score = torch.sigmoid(model(x))
            test_scores.append(score.cpu().numpy())
    return np.concatenate(test_scores)

def load_preproc_and_tokenize():
    train_df = pd.read_csv(path+'train.csv')
    test_df = pd.read_csv(path+'test.csv')

    print('cleaning text...')
    t0 = time.time()
    train_df['comment_text'] = train_df['comment_text'].apply(clean_text)
    test_df['comment_text'] = test_df['comment_text'].apply(clean_text)
    print('cleaning complete in {:.0f} seconds.'.format(time.time()-t0))

    y_train = train_df[label_cols].values.astype('uint8')
    full_text = train_df['comment_text'].tolist() + test_df['comment_text'].tolist()

    print('tokenizing...')
    t0 = time.time()
    word_to_idx, idx_to_word = word_idx_map(full_text, args.vocab_size)
    x_train = tokenize(train_df['comment_text'], word_to_idx, args.maxlen)
    x_test = tokenize(test_df['comment_text'], word_to_idx, args.maxlen)
    print('tokenizing complete in {:.0f} seconds.'.format(time.time()-t0))

    return x_train, y_train, x_test, test_df['id'], word_to_idx


## main()
np.random.seed(SEED)
train_seq, train_tars, x_test, test_id, word_to_idx = load_preproc_and_tokenize()

print('loading embeddings...')
t0 = time.time()
# embed_mat = np.concatenate(
#     [get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], axis=-1)
embed_mat = np.mean([get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], 0)
print('loading complete in {:.0f} seconds.'.format(time.time()-t0))


# preparation
test_preds = []     # for the predictions on the testset
ema_test_preds = [] # for the ema predictions on the testset
test_loader, test_original_indices = prepare_loader(x_test, split='test')

# model setup
models = torch.load(model_path+'models.pt')

# inference
print('start inference...')
if args.enable_ckpt_ensemble:
    ckpt_weights = [2**e for e in range(args.epochs)]
    for i in range(args.nb_models):
        single_test_preds = []
        for e in range(args.epochs):
            model = JigsawNet(*embed_mat.shape, 128, embed_mat)
            model.to(device)
            model.load_state_dict(models[f'fold_{i}_epk_{e}'])
            test_scores = eval_model(model, test_loader)
            single_test_preds.append(test_scores[test_original_indices])
        test_preds.append(np.average(single_test_preds, weights=ckpt_weights, axis=0))
    test_preds = np.mean(test_preds, 0)

if args.ckpt_per_fold:
    for i in range(args.nb_models):
        model = JigsawNet(*embed_mat.shape, 128, embed_mat)
        model.to(device)
        model.load_state_dict(models[f'fold_{i}'])
        test_scores = eval_model(model, test_loader)
        test_preds.append(test_scores[test_original_indices])
    test_preds = np.mean(test_preds, 0)

if args.wavg_per_fold:
    model = JigsawNet(*embed_mat.shape, 128, embed_mat)
    avgd_model = copy.deepcopy(model)
    avgd = NNAverage(avgd_model, 1./args.nb_models)

    for i in range(args.nb_models):
        model.load_state_dict(models[f'fold_{i}'])
        avgd.update(model)
    avgd.set_weights(avgd_model)
    avgd_model.rnns.lstm.flatten_parameters()
    avgd_model.rnns.gru.flatten_parameters()
    avgd_model.to(device)
    test_preds = eval_model(avgd_model, test_loader)[test_original_indices]

# for i in range(args.nb_models):
#     ema_model = JigsawNet(*embed_mat.shape, 128, embed_mat)
#     ema_model.to(device)
#     ema_model.load_state_dict(models[f'ema_fold_{i}'])
#     ema_test_preds.append(eval_model(ema_model, test_loader)[test_original_indices])
# ema_test_preds = np.mean(ema_test_preds, 0)


# submit
# test_preds = 0.6*test_preds + 0.4*ema_test_preds
submission = pd.DataFrame(test_preds, columns=label_cols)
submission['id'] = test_id
submission.to_csv('submission.csv', index=False)