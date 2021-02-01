# word identification code

import numpy as np
import pickle
from scipy.spatial.distance import cosine
import scipy
from numpy import linalg as LA
import random
from tqdm import tqdm
import sys
import argparse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import operator
import sys
import pdb

random.seed(10)

parser = argparse.ArgumentParser()

parser.add_argument("--embed_a")
parser.add_argument("--embed_b")
parser.add_argument("--data_a")
parser.add_argument("--data_b")
parser.add_argument("--name_split_a")
parser.add_argument("--name_split_b")
parser.add_argument("--out_topk")
parser.add_argument("--freq_thr", type=float, default=0.00001, help="frequency threshold")
parser.add_argument("--min_count", type=int, default=200, help="min appearances of a word")
parser.add_argument("--k", type=int, default=500, help="k of k-NN to use")

def load_embeddings_hamilton(filename):
    # loads word embeddings and vocab file from hamilton codebase

    # load vocab file
    print('loading ...')
    vocab_file = open(filename + '-vocab.pkl', 'r')
    data = pickle.load(vocab_file)
    vocab = [line.strip() for line in data]
    w2i = {w: i for i, w in enumerate(vocab)}

    # load word embeddings
    wv = np.load(filename + '-w.npy')

    return vocab, wv, w2i

def load_embeddings_from_np(filename):
    # load word embedding file and vocab file from file_prefix
    print('loading %s'%filename)
    with open(filename + '.vocab', 'r') as f_embed:
        vocab = [line.strip() for line in f_embed]
    w2i = {w: i for i, w in enumerate(vocab)}
    wv = np.load(filename + '.wv.npy')
    return vocab, wv, w2i

def normalize(wv):
    # normalize vectors
    norms = np.apply_along_axis(LA.norm, 1, wv)
    wv = wv / norms[:, np.newaxis]
    return wv

def load_and_normalize(lang, filename, vocab, wv, w2i, hamilton=False):
    # load word embeddings, vocab file and update the global maps (vocab, wv, w2i)

    # load word embeddings, vocab file
    if hamilton:
        vocab_muse, wv_muse, w2i_muse = load_embeddings_hamilton(filename)
    else:
        vocab_muse, wv_muse, w2i_muse = load_embeddings_from_np(filename)

    # normalize the word embeddings
    wv_muse = normalize(wv_muse)

    # update the global maps
    vocab[lang] = vocab_muse
    wv[lang] = wv_muse
    w2i[lang] = w2i_muse
    print('loaded and normalized %s embeddings'%filename)

def create_aligned(w2i, wv, vocab, Q_bef, space1, aligned):
    # update the global maps with the aligned vectors, vocab and word to sequence id mapping
    wv[aligned] = np.zeros(wv[space1].shape)
    for i, vec in enumerate(wv[space1]):
        wv[aligned][i, :] = np.dot(vec, Q_bef)
    vocab[aligned] = vocab[space1]
    w2i[aligned] = w2i[space1]

def align(w2i, wv, vocab, space1, space2, aligned):
    # align the word embeddings from two spaces using orthogonal procrustes (OP)

    # identify the common words in both spaces
    train_words = list(set(vocab[space1]).intersection(set(vocab[space2])))

    # perform OP
    num = len(train_words)
    mat_bef = np.zeros((num, 300))
    mat_aft = np.zeros((num, 300))
    for i, w in enumerate(train_words):
        mat_bef[i, :] = wv[space1][w2i[space1][w]]
        mat_aft[i, :] = wv[space2][w2i[space2][w]]
    Q_bef, _ = scipy.linalg.orthogonal_procrustes(mat_bef, mat_aft)

    # update the global maps
    create_aligned(w2i, wv, vocab, Q_bef, space1, aligned)

    # normalize the aligned embeddings
    wv[aligned] = normalize(wv[aligned])

def topK(w, space, k=10, count = None, min_freq = 0):
    # identify the top k neighbors of a word in a space

    # extract the word vector for word w
    idx = w2i[space][w]
    vec = wv[space][idx, :]

    # compute similarity of w with all words in the vocabulary
    sim = wv[space].dot(vec)

    # sort similarities by descending order
    sort_sim = (sim.argsort())[::-1]

    # choose topK
    if count:
        # print("eheyu")
        # consider only the neighbors whose raw frequency is greater than min_freq
        best = []
        for i in sort_sim:
            if i != idx and count[vocab[space][i]] > min_freq:
                best.append(i)
                if len(best) == k:
                    break
    else:
        best = sort_sim[:(k + 1)]

    return [vocab[space][i] for i in best if i != idx]

def extract_freqs(filename, vocab):
    # raw counts
    print('extracting freqs %s'%filename)
    count = defaultdict(int)
    with open(filename, 'r') as f:
        for l in f:
            for w in l.strip().split():
                count[w] += 1

    # consider only words in the vocabulary
    count_vocab = defaultdict(int)
    for w in vocab:
        if w in count:
            count_vocab[w] = count[w]

    # normalized frequencies
    tot = sum([count_vocab[item] for item in count_vocab])
    freq_norm = defaultdict(int)
    for w in count_vocab:
        freq_norm[w] = count_vocab[w] / float(tot)

    # top-frequent
    top_freq = defaultdict(int)
    sorted_words = [x[0] for x in sorted(count_vocab.items(), key=operator.itemgetter(1))]
    cutoff = len(sorted_words) / float(20)
    top_freq_words = sorted_words[int(4 * cutoff):-200]  # -int(cutoff)]
    for w in top_freq_words:
        top_freq[w] = count[w]

    print('done')
    return freq_norm, count_vocab, top_freq

def load_all_embeddings(args):
    # loads embedding from both spaces and both seeds (123, 456) and perform alignment
    vocab = {}
    wv = {}
    w2i = {}
    load_and_normalize(val1+'0', args.embed_a, vocab, wv, w2i)
    load_and_normalize(val2+'0', args.embed_b, vocab, wv, w2i)
    load_and_normalize(val1+'1', args.embed_a.replace('seed123', 'seed456'), vocab, wv, w2i)
    load_and_normalize(val2+'1', args.embed_b.replace('seed123', 'seed456'), vocab, wv, w2i)
    align(w2i, wv, vocab, val1+'0', val2+'0', val1+'_a0')
    align(w2i, wv, vocab, val1+'1', val2+'1', val1+'_a1')
    return vocab, wv, w2i


def NN_scores(space1, space2, freq1, freq2, count1, count2):
    # compute nearest neighbors overlap for all the common words
    nn_scores = []
    pbar = tqdm(total=len(vocab[space1]))
    for i, w in enumerate(vocab[space1]):
        # if w not in s_words and w in freq1 and w in freq2 and w in AJR and count1[w] > MIN_COUNT and count2[w] > MIN_COUNT:
        if w not in s_words and w in freq1 and w in freq2 and count1[w] > MIN_COUNT and count2[w] > MIN_COUNT:
            neighbors_bef = set(topK(w, space1, args.k, count1, 10))
            neighbors_aft = set(topK(w, space2, args.k, count2, 10))
            # print(neighbors_aft)
            nn_scores.append((len(neighbors_bef.intersection(neighbors_aft)), w))
        if i%10 == 0:
            pbar.update(10)
    pbar.close()
    print('len of ranking', len(nn_scores))

    # rank these words
    nn_scores_sorted = sorted(nn_scores)
    return nn_scores_sorted


def detect(args):
    # extract frequencies
    _, count_vocab_val1, top_freq_val1 = extract_freqs(
        args.data_a, vocab[val1 + '0'])
    _, count_vocab_val2, top_freq_val2 = extract_freqs(
        args.data_b, vocab[val2 + '0'])

    # detect words using nn
    print('detecting words using NN ...')
    nn = []
    for i in range(2):
        print(val1 + '_a' + str(i))
        print(val2 + str(i))
        nn.append(NN_scores(val1 + '_a' + str(i), val2 + str(i), top_freq_val1, top_freq_val2, count_vocab_val1,
                            count_vocab_val2))
    print(nn)


if __name__ == '__main__':
    assert(sys.version_info[0] > 2)
    args = parser.parse_args()
    MIN_COUNT = 0

    s_words = set(stopwords.words('english'))
    val1, val2 = args.name_split_a, args.name_split_b
    vocab, wv, w2i = load_all_embeddings(args)
    detect(args)



