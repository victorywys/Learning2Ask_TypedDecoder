import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope
import numpy as np

def output_projection_layer(num_units, num_symbols, num_qwords, num_samples=None, question_data=True, name="output_projection"):
    def output_fn(outputs, keyword_tensor):
        with variable_scope.variable_scope('%s' % name):
            print "outputs.shape:", outputs.get_shape()
            print "keywrod_tensor.shape:", keyword_tensor.get_shape()
            weights00 = tf.get_variable("weights00", [num_units, 2000])
            bias00 = tf.get_variable("biases00", [2000])    
            weights01 = tf.get_variable("weights01", [2000, 3]) #0:question_words; 1:other words; 2:keywords
            bias01 = tf.get_variable("bias01", [3])
            weights1 = tf.get_variable("weights1", [num_units, num_symbols])
            bias1 = tf.get_variable("biases1", [num_symbols])

            local_outputs = tf.reshape(outputs, [-1, num_units]) #batch * num_units
            print "local_outputs.shape:", local_outputs.get_shape()
            result_p = tf.matmul(tf.matmul(local_outputs, weights00) + bias00, weights01) + bias01
            #gumble softmax
            u = tf.random_uniform(tf.shape(result_p), 1e-10, 1 - 1e-10)
            g = -tf.log(-tf.log(u))
            result_p = (result_p + g*1.2) / tf.constant(0.5)
            result_p = tf.nn.softmax(result_p)
            print "result_p.shape:", result_p.get_shape()

            #generate a mask according to the type distribution over the whole vocabulary
            result_p_2 = tf.reduce_sum(tf.matmul(tf.expand_dims(result_p, axis = 1), keyword_tensor), axis = 1)
            #original distribution over the whole vocabulary distribution
            result_symbol_p = tf.nn.softmax(tf.matmul(local_outputs, weights1) + bias1)
            result_p_3 = result_p_2 * result_symbol_p
            result_sum = tf.reshape(tf.reduce_sum(result_p_3,  axis = 1), [-1,1])
            #final (type-modeled) distribution over the whole vocabulary
            result_p_final = result_p_3 / tf.matmul(result_sum, tf.constant(1.0, shape=[1, num_symbols]))
        return result_p_final

    def sampled_sequence_loss(outputs, targets, masks, keyword_tensor, word_type):
        with variable_scope.variable_scope('decoder/%s' % name):
            weights00 = tf.get_variable("weights00", [num_units, 2000])
            bias00 = tf.get_variable("biases00", [2000])
            weights01 = tf.get_variable("weights01", [2000, 3]) #0:question_words; 1:other words; 2:keywords
            bias01 = tf.get_variable("bias01", [3])
            weights1 = tf.get_variable("weights1", [num_units, num_symbols])
            bias1 = tf.get_variable("biases1", [num_symbols])

            #lcy:reduce dimention
            local_labels = tf.reshape(targets, [-1, 1]) #batch * len * num_units
            local_labels = tf.Print(local_labels, ["local_labels", local_labels, "tf.reduce_sum(local_labels)", tf.reduce_sum(local_labels)])
            print "local_labels.shape:", local_labels.get_shape()
            local_outputs = tf.reshape(outputs, [-1, num_units])
            print "local_outputs.shape:", local_outputs.get_shape()
            local_masks = tf.reshape(masks, [-1]) #batch * len
            if question_data:
                result_p = tf.matmul(tf.matmul(local_outputs, weights00) + bias00, weights01) + bias01

                #gumbel softmax
                u = tf.random_uniform(tf.shape(result_p), 1e-10, 1 - 1e-10)
                g = -tf.log(-tf.log(u))
                result_p = (result_p + g * 0.3) / tf.constant(0.7)
                result_p_softmax = tf.clip_by_value(tf.nn.softmax(result_p), 1e-10, 1.0) #(batch*len) * 3
                print "result_p_softmax.shape:", result_p_softmax.get_shape()

                #calculate type loss
                result_p_labels = tf.one_hot(word_type, 3)
                print "result_p_labels.shape:", result_p_labels.get_shape()
                type_loss = -tf.reduce_sum(result_p_labels * tf.log(result_p_softmax), axis = 1)
                type_loss = type_loss * local_masks
                loss2 = tf.reduce_sum(type_loss)
            else:
                result_p = tf.matmul(local_outputs, tf.zeros([num_units, 3]))
            
            #prepare for sampled softmax
            result_p_2 = tf.reduce_sum(tf.matmul(tf.expand_dims(result_p, axis = 1), keyword_tensor), axis = 1)
            result_p_2 = tf.Print(result_p_2, ["result_p_2", result_p_2])
            result_symbol = tf.matmul(local_outputs, weights1) + bias1
            result_before_softmax = result_p_2 + result_symbol#(batch*len) * numsymbols

            #calculate sampled softmax
            #local_label : (batch_size * len) * 1
            local_labels = tf.cast(local_labels, tf.int64)
            print "local_labels:", local_labels.get_shape()
            sampled_values = tf.nn.log_uniform_candidate_sampler(
                    true_classes=local_labels,
                    num_true=1,
                    num_sampled = num_samples,
                    unique=True,
                    range_max=num_symbols
                    )
            sampled, true_expected_count, sampled_expected_count = (tf.stop_gradient(s) for s in sampled_values)
            out_logits_sampled = tf.transpose(tf.gather(tf.transpose(result_before_softmax), sampled))
            out_logits_labels = tf.expand_dims(tf.reduce_sum(result_before_softmax * tf.one_hot(tf.reduce_sum(local_labels, axis = 1), num_symbols), axis = 1), axis = 1)
            out_logits = tf.concat([out_logits_labels, out_logits_sampled], 1)
            out_labels = tf.concat([tf.ones_like(out_logits_labels, tf.float32), tf.zeros_like(out_logits_sampled, tf.float32)], 1)

            vocab_p_softmax = tf.clip_by_value(tf.nn.softmax(out_logits), 1e-10, 1.0)
            #calculate sampled softmax loss 
            local_loss = -tf.reduce_sum(out_labels * tf.log(vocab_p_softmax), axis = 1)
            print "local_loss1:", local_loss.get_shape()
            print "local_masks.shape:", local_masks.get_shape()
            local_loss = local_loss * local_masks
            print "local_loss2:", local_loss.get_shape()
            local_loss = tf.Print(local_loss, ["local loss", local_loss], summarize = 5)
            loss = tf.reduce_sum(local_loss)
            print "loss:", loss, "loss.shape:", loss.get_shape()

            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            if question_data:
                return (tf.constant(0.8) * loss + loss2) / total_size, loss / total_size
            else:
                return loss / total_size

    return output_fn, sampled_sequence_loss

