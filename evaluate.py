# -*- coding: utf-8 -*-
# @Author: fducau
# @Date:   2016-11-23 13:50:50
# @Last Modified by:   fducau
# @Last Modified time: 2016-11-27 19:38:17
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

from scipy import optimize, stats
from collections import OrderedDict

import wmt14enfr
import iwslt14zhen
import openmt15zhen
import trans_enhi
import stan
import data_iterator

import dateutil
import dateutil.tz
import datetime

from nmt import *

def load_lines(input_file):
    x = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            line_w_eos = line + ' 0'
            source = numpy.array(map(int, line_w_eos.split()), dtype=numpy.int64)

            x.append(source)
    return x

def evaluate(dim_word=128,
             dim=512,
             encoder='gru',
             decoder='gru_cond',
             hiero=None,  # 'gru_hiero', # or None
             patience=5,
             max_epochs=100,
             dispFreq=250,
             decay_c=0.,
             alpha_c=0.,
             diag_c=0.,
             lrate=0.01,
             n_words_src=20000,
             n_words=20000,
             maxlen=50,
             optimizer='adadelta',
             batch_size=100,
             valid_batch_size=100,
             saveto='ckt',
             validFreq=50,
             saveFreq=50,
             sampleFreq=150,
             dataset='data_iterator',
             dictionary='./data/en_dict.pkl',
             dictionary_src='./data/ja_dict.pkl',
             use_dropout=False,
             correlation_coeff=0.1,
             clip_c=1.,
             input_file='./chat_files/input_sentences.txt_idx'):
    
    source_utterances = load_lines(input_file)
    # Model options
    model_options = locals().copy()

    if dictionary:
        with open(dictionary, 'rb') as f:
            word_dict = pkl.load(f)
        word_idict = dict()
        for kk, vv in word_dict.iteritems():
            word_idict[vv] = kk

    if dictionary_src:
        with open(dictionary_src, 'rb') as f:
            word_dict_src = pkl.load(f)
        word_idict_src = dict()
        for kk, vv in word_dict_src.iteritems():
            word_idict_src[vv] = kk

    # Reload previous saved options
    if model:
        with open('{}.npz.pkl'.format(model), 'rb') as f:
            models_options = pkl.load(f)
    else:
        raise ValueError('No model specified')

    print 'Building model...'
    params = init_params(model_options)
    # reload parameters
    if model:
        params = load_params(model, params)

    tparams = init_tparams(params)

    trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost = build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    # theano.printing.debugprint(cost.mean(), file=open('cost.txt', 'w'))

    print 'Buliding sampler...'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    history_errs = []
    # reload history
    if model and os.path.exists(model):
        history_errs = list(numpy.load(model)['history_errs'])
    best_p = None
    bad_count = 0



    uidx = 0
    estop = False
    save_turn = 0
    ####################
    # Main training loop
    ####################


    for i, x in enumerate(source_utterances):
        stochastic = False
        sample, score = gen_sample(tparams, f_init, f_next, x[:, None],
                                   model_options, trng=trng, k=1, maxlen=30,
                                   stochastic=stochastic, argmax=True)

        print('Source {}: '.format(i) + print_utterance(x, word_idict))
        if stochastic:
            ss = sample
        else:
            score = score / numpy.array([len(s) for s in sample])
            ss = sample[score.argmin()]
        print('Sample {}:'.format(i) + print_utterance(ss, word_idict))





if __name__ == '__main__':
    evaluate(model='./chat_files/ckt_noatt_001_MED1_turn_0')

