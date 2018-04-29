#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np

def multi_bidirectional_rnn(cell, num_units, num_layers, inputs, seq_lengths):
    """
    paramaters:
    cell: rnn cell class
    num_units: hidden units of RNN cell
    num_layers: the number of Rnn layers
    inputs: the input sequence
    seq_lengths: the length of the intput sequence
    
    return the output of the last layer after concating
    """
    _inputs = inputs
    _state = []
    for _ in range(num_layers):
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            rnn_cell_fw = cell(num_units)
            rnn_cell_bw = cell(num_units)
            output, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths, dtype=tf.float32)

            _inputs = tf.concat(output, 2)
            _state.append(tf.concat(state, 1))

    return _inputs, tuple(_state)
