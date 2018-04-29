import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell, LSTMCell, MultiRNNCell
import attention_decoder_fn
from tensorflow.contrib.seq2seq.python.ops.seq2seq import dynamic_rnn_decoder
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from output_projection import output_projection_layer
from bidirectional_dynamic_rnn import multi_bidirectional_rnn

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class Seq2SeqModel(object):
    def __init__(self,
            num_symbols,
            num_qwords, #modify
            num_embed_units,
            num_units,
            num_layers,
            is_train,
            vocab=None,
            embed=None,
            question_data=True,
            learning_rate=0.5,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            use_lstm=False,
            use_bidrnn=False):

        self.posts = tf.placeholder(tf.string, shape=(None, None))  # batch*len
        self.posts_length = tf.placeholder(tf.int32, shape=(None))  # batch
        self.responses = tf.placeholder(tf.string, shape=(None, None))  # batch*len
        self.responses_length = tf.placeholder(tf.int32, shape=(None))  # batch
        self.keyword_tensor = tf.placeholder(tf.float32, shape=(None, 3, None)) #(batch * len) * 3 * numsymbol, not used in STD
        self.word_type = tf.placeholder(tf.int32, shape=(None))   #(batch * len)

        # build the vocab table (string to index)
        if is_train:
            self.symbols = tf.Variable(vocab, trainable=False, name="symbols")
        else:
            self.symbols = tf.Variable(np.array(['.']*num_symbols), name="symbols")
        self.symbol2index = HashTable(KeyValueTensorInitializer(self.symbols,
            tf.Variable(np.array([i for i in range(num_symbols)], dtype=np.int32), False)),
            default_value=UNK_ID, name="symbol2index")
        #string2index for post and response
        self.posts_input = self.symbol2index.lookup(self.posts)   # batch*len
        self.responses_target = self.symbol2index.lookup(self.responses)   #batch*len
        
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        self.responses_input = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32)*GO_ID,
            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1)   # batch*len
        #delete the last column of responses_target) and add 'GO at the front of it.
        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1,
            decoder_len), reverse=True, axis=1), [-1, decoder_len]) #bacth * len

        print "embedding..."
        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
            #self.embed = tf.Print(self.embed, ['embed', self.embed])

        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts_input) #batch*len*unit
        self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input)

        print "embedding finished"

        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units)] * num_layers)
        else:
            cell = MultiRNNCell([GRUCell(num_units)] * num_layers)

        #for bidirectional rnn, not used in STD in final experiment
        if use_bidrnn:
            if use_lstm:
                encoder_cell = LSTMCell
            else:
                encoder_cell = GRUCell
            
            # rnn encoder
            encoder_output, encoder_state = multi_bidirectional_rnn(encoder_cell, num_units / 2, num_layers, self.encoder_input, self.posts_length)
        else:
            # rnn encoder
            encoder_output, encoder_state = dynamic_rnn(cell, self.encoder_input,
                    self.posts_length, dtype=tf.float32, scope="encoder")
        
            # get output projection function
        output_fn, sampled_sequence_loss = output_projection_layer(num_units,
                num_symbols, num_qwords, num_samples, question_data)

        print "encoder_output.shape:", encoder_output.get_shape()

        # get attention function
        attention_keys, attention_values, attention_score_fn, attention_construct_fn \
              = attention_decoder_fn.prepare_attention(encoder_output, 'luong', num_units)

        # get decoding loop function
        decoder_fn_train = attention_decoder_fn.attention_decoder_fn_train(encoder_state,
                attention_keys, attention_values, attention_score_fn, attention_construct_fn)
        decoder_fn_inference = attention_decoder_fn.attention_decoder_fn_inference(output_fn,
                self.keyword_tensor,
                encoder_state, attention_keys, attention_values, attention_score_fn,
                attention_construct_fn, self.embed, GO_ID, EOS_ID, max_length, num_symbols)

        if is_train:
            # rnn decoder
            self.decoder_output, _, _ = dynamic_rnn_decoder(cell, decoder_fn_train,
                    self.decoder_input, self.responses_length, scope="decoder")
            # calculate the loss of decoder
            self.decoder_loss, self.ppl_loss = sampled_sequence_loss(self.decoder_output,
                    self.responses_target, self.decoder_mask, self.keyword_tensor, self.word_type)

            # building graph finished and get all parameters
            self.params = tf.trainable_variables()

            for item in tf.trainable_variables():
                print item.name, item.get_shape()

            # initialize the training process
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                    dtype=tf.float32)
            self.learning_rate_decay_op = self.learning_rate.assign(
                    self.learning_rate * learning_rate_decay_factor)

            self.global_step = tf.Variable(0, trainable=False)

            # calculate the gradient of parameters
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients = tf.gradients(self.decoder_loss, self.params)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
                    max_gradient_norm)
            self.update = opt.apply_gradients(zip(clipped_gradients, self.params),
                    global_step=self.global_step)
             
            #self.train_op = tf.train.AdamOptimizer().minimize(self.decoder_loss, global_step=self.global_step)

        else:
            # rnn decoder
            self.decoder_distribution, _, _ = dynamic_rnn_decoder(cell, decoder_fn_inference,
                    scope="decoder")
            print("self.decoder_distribution.shape():",self.decoder_distribution.get_shape())
            self.decoder_distribution = tf.Print(self.decoder_distribution, ["distribution.shape()", tf.reduce_sum(self.decoder_distribution)])
            # generating the response
            self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
                [2, num_symbols-2], 2)[1], 2) + 2 # for removing UNK
            self.generation = tf.nn.embedding_lookup(self.symbols, self.generation_index)

            self.params = tf.trainable_variables()

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def step_decoder(self, session, data, forward_only=False):
        input_feed = {self.posts: data['posts'],
                self.posts_length: data['posts_length'],
                self.responses: data['responses'],
                self.responses_length: data['responses_length'],
                self.keyword_tensor: data['keyword_tensor'],      #keyword_tensor not used in STD
                self.word_type: data['word_type']
                }
        if forward_only:
            output_feed = [self.decoder_loss, self.ppl_loss]
        else:
            output_feed = [self.decoder_loss, self.gradient_norm, self.update]

        return session.run(output_feed, input_feed)

    def inference(self, session, data):
        input_feed = {self.posts: data['posts'], self.posts_length: data['posts_length'], self.keyword_tensor: data['keyword_tensor']}
        output_feed = [self.generation]
        return session.run(output_feed, input_feed)
