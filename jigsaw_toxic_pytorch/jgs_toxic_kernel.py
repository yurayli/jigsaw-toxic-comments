# avg 3 embs, 1-cycle
import os, time, json, re, copy
import itertools, argparse, pickle, random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Sampler

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

SEED = 2019
path = '../input/jigsaw-toxic-comment-classification-challenge/'
output_path = './'
EMBEDDING_FILES = [
    '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl',
    # '../input/pickled-paragram-300-vectors-sl999/paragram_300_sl999.pkl',
    '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
]

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=220,
                    help='maximum length of a input sequence')
parser.add_argument('--vocab-size', type=int, default=100000)
parser.add_argument('--n-splits', type=int, default=5,
                    help='splits of n-fold cross validation')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--enable-ckpt-ensemble', type=bool, default=1)
parser.add_argument('--ckpt-per-fold', type=bool, default=1)
args = parser.parse_args()

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

    return np.array([text_to_id(c, word_to_idx, maxlen) for c in comments])


# Seed for randomness
def seed_torch(seed=SEED):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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


# one-cycle scheduler
class OneCycleScheduler(object):

    def __init__(self, optimizer, epochs, train_loader, max_lr=3e-3,
                 moms=(.95, .85), div_factor=25, sep_ratio=0.3, final_div=None):

        self.optimizer = optimizer

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
            self.init_lrs = [lr/div_factor for lr in self.max_lrs]
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
            self.init_lrs = [max_lr/div_factor] * len(optimizer.param_groups)

        self.final_div = final_div
        if self.final_div is None: self.final_div = div_factor*1e4
        self.final_lrs = [lr/self.final_div for lr in self.max_lrs]
        self.moms = moms

        self.total_iteration = epochs * len(train_loader)
        self.up_iteration = int(self.total_iteration * sep_ratio)
        self.down_iteration = self.total_iteration - self.up_iteration

        self.curr_iter = 0
        self._assign_lr_mom(self.init_lrs, [moms[0]]*len(optimizer.param_groups))

    def _assign_lr_mom(self, lrs, moms):
        for param_group, lr, mom in zip(self.optimizer.param_groups, lrs, moms):
            param_group['lr'] = lr
            param_group['betas'] = (mom, 0.999)

    def _annealing_cos(self, start, end, pct):
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start-end)/2 * cos_out

    def step(self):
        self.curr_iter += 1

        if self.curr_iter <= self.up_iteration:
            pct = self.curr_iter / self.up_iteration
            curr_lrs = [self._annealing_cos(min_lr, max_lr, pct) \
                            for min_lr, max_lr in zip(self.init_lrs, self.max_lrs)]
            curr_moms = [self._annealing_cos(self.moms[0], self.moms[1], pct) \
                            for _ in range(len(self.optimizer.param_groups))]
        else:
            pct = (self.curr_iter-self.up_iteration) / self.down_iteration
            curr_lrs = [self._annealing_cos(max_lr, final_lr, pct) \
                            for max_lr, final_lr in zip(self.max_lrs, self.final_lrs)]
            curr_moms = [self._annealing_cos(self.moms[1], self.moms[0], pct) \
                            for _ in range(len(self.optimizer.param_groups))]

        self._assign_lr_mom(curr_lrs, curr_moms)


# solver of model with validation
class NetSolver(object):

    def __init__(self, model, optimizer, scheduler=None, val_indices=None, checkpoint_name='toxic_comment'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_name = checkpoint_name

        self.models = {}
        self.val_indices = val_indices
        if args.enable_ckpt_ensemble:
            self.single_val_preds = []

        self.model = self.model.to(device=device)
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        self.best_val_loss = 0.
        self.best_val_auc = 0.
        self.loss_history = []
        self.val_loss_history = []
        self.auc_history = []
        self.val_auc_history = []

    def _save_checkpoint(self, epoch, l_val, a_val, save_trn_states=False):
        torch.save(self.model.state_dict(),
            output_path+self.checkpoint_name+'_%.3f_%.3f_epoch_%d.pth.tar' %(l_val, a_val, epoch))
        if save_trn_states:
            checkpoint = {
                'optimizer': str(type(self.optimizer)),
                'scheduler': str(type(self.scheduler)),
                'epoch': epoch,
            }
            with open(output_path+'hyper_param_optim.json', 'w') as f:
                json.dump(checkpoint, f)


    def forward_pass(self, x, y):
        x = x.to(device=device, dtype=torch.long)
        y = y.to(device=device, dtype=dtype)
        scores = self.model(x)
        loss = F.binary_cross_entropy_with_logits(scores, y)
        return loss, torch.sigmoid(scores)


    def lr_range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        epochs = int(np.ceil(num_it/len(train_loader)))
        n_groups = len(self.optimizer.param_groups)

        if isinstance(start_lr, list) or isinstance(start_lr, tuple):
            if len(start_lr) != n_groups:
                raise ValueError("expected {} max_lr, got {}".format(n_groups, len(start_lr)))
            self.start_lrs = list(start_lr)
        else:
            self.start_lrs = [start_lr] * n_groups

        curr_lrs = self.start_lrs*1
        for param_group, lr in zip(self.optimizer.param_groups, curr_lrs):
            param_group['lr'] = lr

        n, lrs_log, loss_log = 0, [], []

        for e in range(epochs):
            self.model.train()
            for x, y in train_loader:
                loss, _ = self.forward_pass(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lrs_log.append(curr_lrs[-1])
                loss_log.append(loss.item())

                # update best loss
                if n == 0:
                    best_loss, n_best = loss.item(), n
                else:
                    if loss.item() < best_loss:
                        best_loss, n_best = loss.item(), n

                # update lr per iter
                n += 1
                curr_lrs = [lr * (end_lr/lr) ** (n/num_it) for lr in self.start_lrs]
                for param_group, lr in zip(self.optimizer.param_groups, curr_lrs):
                    param_group['lr'] = lr

                # stopping condition
                if n == num_it or (stop_div and (loss.item() > 4*best_loss or torch.isnan(loss))):
                    break

        print('minimum loss {}, at lr {}'.format(best_loss, lrs_log[n_best]))
        return lrs_log, loss_log


    def train(self, loaders, epochs, i=None):
        train_loader, val_loader = loaders

        # start training for epochs
        for e in range(epochs):
            self.model.train()
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            running_loss = 0.

            for x, y in train_loader:
                loss, _ = self.forward_pass(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * x.size(0)

            N = len(train_loader.dataset)
            train_loss = running_loss / N
            train_auc, _l, _s = self.check_auc(train_loader, num_batches=50)
            val_auc, val_loss, val_scores = self.check_auc(val_loader, save_scores=True)
            val_scores = val_scores[self.val_indices]

            self.log_and_checkpoint(i, e, train_loss, val_loss, train_auc, val_auc, val_scores)

            if self.scheduler:
                self.scheduler.step()


    def train_one_cycle(self, loaders, epochs, ema=None, i=None):
        train_loader, val_loader = loaders

        # start training for epochs
        for e in range(epochs):
            self.model.train()
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            running_loss = 0.

            for x, y in train_loader:
                loss, _ = self.forward_pass(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * x.size(0)

                # update lr, mom per iter
                if self.scheduler:
                    self.scheduler.step()

                if ema:
                    ema.on_batch_end(self.model)

            N = len(train_loader.dataset)
            train_loss = running_loss / N
            train_auc, _l, _s = self.check_auc(train_loader, num_batches=50)
            val_auc, val_loss, val_scores = self.check_auc(val_loader, save_scores=True)
            val_scores = val_scores[self.val_indices]

            self.log_and_checkpoint(i, e, train_loss, val_loss, train_auc, val_auc, val_scores)


    def log_and_checkpoint(self, i, e, train_loss, val_loss, train_auc, val_auc, val_scores):
        # checkpoint and record/print metrics at epoch end
        self.loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.auc_history.append(train_auc)
        self.val_auc_history.append(val_auc)

        # print logs
        print('{"metric": "AUC", "value": %.4f, "epoch": %d}' % (train_auc, e+1))
        print('{"metric": "Val. AUC", "value": %.4f, "epoch": %d}' % (val_auc, e+1))
        print('{"metric": "Loss", "value": %.4f, "epoch": %d}' % (train_loss, e+1))
        print('{"metric": "Val. Loss", "value": %.4f, "epoch": %d}' % (val_loss, e+1))

        if args.enable_ckpt_ensemble:
            self.single_val_preds.append(val_scores)
            self.models[f'fold_{i}_epk_{e}'] = self.model.state_dict()

        if e == 0:
            self.best_val_auc = val_auc
            self.best_val_loss = val_loss
        if val_auc > self.best_val_auc:
            print('updating best val auc...')
            self.best_val_auc = val_auc
            if args.ckpt_per_fold:
                self.models[f'fold_{i}'] = self.model.state_dict()
                self.best_val_scores = val_scores
        if val_loss < self.best_val_loss:
            print('updating best val loss...')
            self.best_val_loss = val_loss
        print()


    def check_auc(self, loader, num_batches=None, save_scores=False):
        self.model.eval()
        targets, scores, losses = [], [], []

        with torch.no_grad():
            for t, (x, y) in enumerate(loader):
                l, score = self.forward_pass(x, y)
                targets.append(y.cpu().numpy())
                scores.append(score.cpu().numpy())
                losses.append(l.item())
                if num_batches is not None and (t+1) == num_batches:
                    break

        targets = np.concatenate(targets)
        scores = np.concatenate(scores)
        auc = roc_auc_score(targets, scores)
        loss = np.mean(losses)

        return auc, loss, scores


# model
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

class WeightEMA(object):
    def __init__(self, model, mu=0.95, sample_rate=1):
        # self.ema_model = copy.deepcopy(model)
        self.mu = mu
        self.sample_rate = sample_rate
        self.sample_cnt = sample_rate
        self.weight_copy = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.weight_copy[name] = (1 - mu) * param.data

    def _update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.mu) * param.data + self.mu * self.weight_copy[name]
                self.weight_copy[name] = new_average.clone()

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if param.requires_grad:
                param.data = self.weight_copy[name]

    def on_batch_end(self, model):
        self.sample_cnt -= 1
        if self.sample_cnt == 0:
            self._update(model)
            self.sample_cnt = self.sample_rate

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

def model_optimizer_init(nb_neurons, embed_mat, ft_lrs):
    model = JigsawNet(*embed_mat.shape, nb_neurons, embed_mat)

    params_emb = [p for p in model.emb_layer.parameters()]
    params_rnn = [p for p in model.rnns.parameters()]
    params_cls = [p for p in model.classifier.parameters()]

    optimizer = torch.optim.Adam(params=[{'params': params_emb, 'lr': ft_lrs[0]}])
    optimizer.add_param_group({'params':params_rnn, 'lr': ft_lrs[1]})
    optimizer.add_param_group({'params':params_cls, 'lr': ft_lrs[2]})

    return model, optimizer


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


def train_val_split(train_x):
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    cv_indices = [(tr_idx, val_idx) for tr_idx, val_idx in kf.split(train_x)]
    return cv_indices


def main(args):
    # load, preprocess, and tokenize data
    train_seq, train_tars, x_test, test_id, word_to_idx = load_preproc_and_tokenize()

    # load pretrained embedding
    print('loading embeddings...')
    t0 = time.time()
    # embed_mat = np.concatenate(
    #     [get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], axis=-1)
    embed_mat = np.mean([get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], 0)
    print('loading complete in {:.0f} seconds.'.format(time.time()-t0))

    # training preparation
    fold_val_preds = []     # oof predictions for ensemble of folds
    ckpt_val_preds = []     # oof predictions for ensemble of ckpts
    ema_val_preds = []      # oof predictions for ensemble from ema of weights
    oof_tars = []           # for the oof targets
    # test_preds = []         # for the predictions on the testset
    # test_loader, test_original_indices = prepare_loader(x_test, split='test')
    cv_indices = train_val_split(train_seq)

    print()
    for i, (trn_idx, val_idx) in enumerate(cv_indices):
        print(f'Fold {i + 1}')

        # train/val split
        x_train, x_val = train_seq[trn_idx], train_seq[val_idx]
        y_train, y_val = train_tars[trn_idx], train_tars[val_idx]
        train_loader = prepare_loader(x_train, y_train, args.batch_size, split='train')
        val_loader, val_original_indices = prepare_loader(x_val, y_val, split='valid')
        oof_tars.append(y_val)

        # model setup
        seed_torch(SEED+i)

        model = JigsawNet(*embed_mat.shape, 128, embed_mat)
        for name, param in model.named_parameters():
           if 'emb' in name:
               param.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        # ft_lrs = [args.lr*0.08, args.lr, args.lr]
        # model, optimizer = model_optimizer_init(128, embed_mat, ft_lrs)
        scheduler = OneCycleScheduler(optimizer, args.epochs, train_loader,
                                      max_lr=args.lr, moms=(.8, .7))
        if i==0:
            solver = NetSolver(model, optimizer, scheduler, val_original_indices)
        else:
            solver.model = model.to(device)
            solver.optimizer = optimizer
            solver.scheduler = scheduler
            solver.val_indices = val_original_indices
            solver.single_val_preds = []
            solver._reset()

        ema_model = copy.deepcopy(model)
        ema = WeightEMA(model, sample_rate=50)

        # train
        t0 = time.time()
        solver.train_one_cycle((train_loader, val_loader), args.epochs, ema, i)
        time_elapsed = time.time() - t0
        print('time = {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

        # inference
        if args.ckpt_per_fold:
            fold_val_preds.append(solver.best_val_scores)

        if args.enable_ckpt_ensemble:
            ckpt_weights = [2**e for e in range(args.epochs)]

            val_scores = np.average(solver.single_val_preds, weights=ckpt_weights, axis=0)
            val_auc = roc_auc_score(y_val, val_scores)
            ckpt_val_preds.append(val_scores)
            print('{"metric": "Ckpt CV Val. AUC", "value": %.4f}' % (val_auc,))

        ema.set_weights(ema_model)
        # https://stackoverflow.com/questions/53231571/what-does-flatten-parameters-do
        ema_model.rnns.lstm.flatten_parameters()
        ema_model.rnns.gru.flatten_parameters()
        solver.models[f'ema_fold_{i}'] = ema_model.state_dict()

        val_scores = eval_model(ema_model.to(device), val_loader, 'val')[val_original_indices]
        val_auc = roc_auc_score(y_val, val_scores)
        ema_val_preds.append(val_scores)
        print('{"metric": "EMA Val. AUC", "value": %.4f}' % (val_auc,))

        # test_scores = eval_model(solver.model, test_loader)[test_original_indices]

        print()

    torch.save(solver.models, 'models.pt')

    # total set validation auc
    fold_val_preds = np.concatenate(fold_val_preds)
    ckpt_val_preds = np.concatenate(ckpt_val_preds)
    ema_val_preds = np.concatenate(ema_val_preds)
    mix_val_preds = np.mean((ckpt_val_preds, ema_val_preds), 0)

    oof_tars = np.concatenate(oof_tars)
    fold_cv_auc = roc_auc_score(oof_tars, fold_val_preds)
    ckpt_cv_auc = roc_auc_score(oof_tars, ckpt_val_preds)
    ema_auc = roc_auc_score(oof_tars, ema_val_preds)
    mix_auc = roc_auc_score(oof_tars, mix_val_preds)

    print(f'For whole oof set, fold cv val auc score: {fold_cv_auc}')
    print(f'For whole oof set, ckpt cv val auc score: {ckpt_cv_auc}')
    print(f'For whole oof set, ema val auc score: {ema_auc}')
    print(f'For whole oof set, mix val auc score: {mix_auc}')

    # submit
    # submit = pd.DataFrame(test_preds, columns=label_cols)
    # submit['id'] = test_id
    # submit.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main(args)