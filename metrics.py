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

#from nltk.translate import bleu
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


# def compute_BLEU(iterator, options, tparams, f_init, f_next, trng, stochastic=False):
# 	bleu_scores = []
# 	for x, y in iterator:
# 		x, x_mask, y, y_mask = prepare_data(x, y, maxlen=100,
# 											n_words_src=options['n_words_src'],
# 											n_words=options['n_words'])
# 
# 
# 		for utterance_idx in xrange(x.shape[1]):
# 			hypothesis, score = gen_sample(tparams, f_init, f_next, x[:, utterance_idx][:, None],
#                            			   model_options, trng=trng, k=1, maxlen=30,
#                                        stochastic=stochastic, argmax=True)
# 
# 			original = y[:, utterance_idx]
# 			hypothesis = ' '.join([str(i) for i in sample])
# 			original = ' '.join([str(i) for i in original])
# 
# 			bleu_scores.append(bleu(original, hypothesis))
# 
# 	bleu_scores = np.array(bleu_scores)
# 	return bleu_scores.mean()