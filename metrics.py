'''
Metrics to evaluate the models
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

import os
import warnings
import sys
import time

import dateutil
import dateutil.tz
import datetime

from data_iterator import prepare_data

from nltk.translate.bleu_score import corpus_bleu
# https://github.com/ddahlmeier/neural_lm/blob/master/lbl.py

def pred_perplexity(f_log_probs, prepare_data, options, iterator, verbose=False):
    log_p = []

    n_done = 0

    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y, maxlen=100, n_words_src=options['n_words_src'], n_words=options['n_words'])
        
        if x == None:
            continue

        batch_log_probs = f_log_probs(x,x_mask,y,y_mask)
        for p in batch_log_probs:
            log_p.append(p)

        if verbose:
            print >>sys.stderr, '%d samples computed'%(n_done)

    perplexity_exponent = np.mean(log_probs)
    return np.exp(perplexity_exponent)


def perplexity_from_logprobs(minus_log_probs):
    return np.exp(np.mean(minus_log_probs))

def compute_BLEU(iterator, options, tparams, f_init, 
                 f_next, generator, trng, stochastic=False):

    hypothesis = []
    originals = []
    for x, y in iterator:
        x, x_mask, y, y_mask = prepare_data(x, y, maxlen=100,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        for utterance_idx in xrange(x.shape[1]):
            h, score = generator(tparams, f_init, f_next, x[:, utterance_idx][:, None],
                                  options, trng=trng, k=1, maxlen=30,
                                  stochastic=stochastic, argmax=True)

            original = y[:, utterance_idx][:, None] 
            original = [i[0] for i in original if i[0]!=0]
            original = [str(i) for i in original]                       

            h = h[0]
            h = [str(i) for i in h]
            
            hypothesis.append(h)
            originals.append(original)

    print originals[0]
    print hypothesis[0]
    return corpus_bleu(originals, hypothesis)
