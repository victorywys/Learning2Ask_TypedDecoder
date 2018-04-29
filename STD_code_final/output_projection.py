import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope
import numpy as np

def output_projection_layer(num_units, num_symbols, num_qwords, num_samples=None, question_data=True, name="output_projection"):
    def output_fn(outputs, keyword_tensor):
        with variable_scope.variable_scope('%s' % name):
            print "outputs.shape:", outputs.get_shape()
            print "keywrod_tensor.shape:", keyword_tensor.get_shape()
            #set trainalbe variable
            weights00 = tf.get_variable("weights00", [num_units, 2000])
            bias00 = tf.get_variable("biases00", [2000])
            weights01 = tf.get_variable("weights01", [2000, 3]) #0:question_words; 1:other words; 2:keywords
            bias01 = tf.get_variable("bias01", [3])
            weights1 = tf.get_variable("weights1", [num_units, num_symbols])
            bias1 = tf.get_variable("biases1", [num_symbols])
            weights2 = tf.get_variable("weights2", [num_units, num_symbols])
            bias2 = tf.get_variable("biases2", [num_symbols])
            weights3 = tf.get_variable("weights3", [num_units, num_symbols])
            bias3 = tf.get_variable("biases3", [num_symbols])
           
            local_outputs = tf.reshape(outputs, [-1, num_units])
            local_outputs = tf.Print(local_outputs, ["local_outputs", local_outputs])
            
            #calculate type distribution
            result_p = tf.matmul(tf.matmul(local_outputs, weights00) + bias00, weights01) + bias01
            result_p_softmax = tf.clip_by_value(tf.nn.softmax(result_p), 1e-10, 1.0) #batch * 3
            result_p_softmax = tf.Print(result_p_softmax, ["result_p_softmax", result_p_softmax, "result_p_sum", tf.
                    reduce_sum(result_p_softmax, axis = 1)], summarize=6)
            print "result_p_softmax.shape:", result_p_softmax.get_shape()
            result_p_expand = tf.matmul(tf.expand_dims(result_p_softmax, axis=2), tf.ones([tf.shape(result_p_softmax)[0], 1, num_symbols],tf.float32))#batch * 3 * num_symbols
            print "result_p_expand.shape:", result_p_expand.get_shape()
            
            #calculate type specific distribution over the whole vocabulary
            result_symbol_1 = tf.nn.softmax(tf.matmul(local_outputs, weights1) + bias1)
            result_symbol_2 = tf.nn.softmax(tf.matmul(local_outputs, weights2) + bias2)
            result_symbol_3 = tf.nn.softmax(tf.matmul(local_outputs, weights3) + bias3)#(batch) * numsymbols
            result_symbol_1 = tf.expand_dims(result_symbol_1, axis=1)
            result_symbol_2 = tf.expand_dims(result_symbol_2, axis=1)
            result_symbol_3 = tf.expand_dims(result_symbol_3, axis=1)
            result_symbol = tf.concat([result_symbol_1, result_symbol_2, result_symbol_3], axis=1)#batch * 3 * num_symbols
            print "result_symbol.shape:", result_symbol.get_shape()

            #calculate hybird distribution over the whole vocabulary
            result_final = tf.clip_by_value(tf.reduce_sum(result_symbol * result_p_expand, axis=1), 1e-10, 1)
            result_final = tf.Print(result_final, ["result_final", result_final, "result_final_sum", tf.reduce_sum(result_final, axis=1)])
            print "result_final", result_final.get_shape()
          
        return result_final

    def sampled_sequence_loss(outputs, targets, masks, keyword_tensor, word_type):
        with variable_scope.variable_scope('decoder/%s' % name):
            weights00 = tf.get_variable("weights00", [num_units, 2000])
            bias00 = tf.get_variable("biases00", [2000])
            weights01 = tf.get_variable("weights01", [2000, 3])  #0:question_words; 1:other words; 2:keywords
            bias01 = tf.get_variable("bias01", [3])
            weights1 = tf.get_variable("weights1", [num_units, num_symbols])
            bias1 = tf.get_variable("biases1", [num_symbols])
            weights2 = tf.get_variable("weights2", [num_units, num_symbols])
            bias2 = tf.get_variable("biases2", [num_symbols])
            weights3 = tf.get_variable("weights3", [num_units, num_symbols])
            bias3 = tf.get_variable("biases3", [num_symbols])
           

            #flat the output and label
            local_labels = tf.reshape(targets, [-1, 1]) 
            local_labels = tf.Print(local_labels, ["local_labels", local_labels, "tf.reduce_sum(local_labels)", tf.reduce_sum(local_labels)])
            print "local_labels.shape:", local_labels.get_shape()
            local_outputs = tf.reshape(outputs, [-1, num_units])  #(batch * len) * num_units
            local_outputs = tf.Print(local_outputs, ["local_outputs", local_outputs])
            print "local_outputs.shape:", local_outputs.get_shape()
            local_masks = tf.reshape(masks, [-1])
                
            #calculate type generation    
            result_p = tf.matmul(tf.matmul(local_outputs, weights00) + bias00, weights01) + bias01
            result_p_softmax = tf.clip_by_value(tf.nn.softmax(result_p), 1e-10, 1.0) #(batch*len) * 3
            result_p_softmax = tf.Print(result_p_softmax, ["result_p_softmax", result_p_softmax, "result_p_sum", tf.
                    reduce_sum(result_p_softmax, axis = 1)], summarize=6)
            print "result_p_softmax.shape:", result_p_softmax.get_shape()

            #calculate type loss
            result_p_labels = tf.one_hot(word_type, 3)
            print "result_p_labels.shape:", result_p_labels.get_shape()
            result_p_labels = tf.Print(result_p_labels, ["result_p_labels", result_p_labels], summarize = 6)
            type_loss = -tf.reduce_sum(result_p_labels * tf.log(result_p_softmax), axis = 1)
            type_loss = type_loss * local_masks
            type_loss = tf.Print(type_loss, ["type_loss", type_loss], summarize = 6)
            loss2 = tf.reduce_sum(type_loss)
            
            #calculate type specific distribution
            result_p_expand = tf.matmul(tf.expand_dims(result_p_labels, axis=2), tf.ones([tf.shape(result_p_softmax)[0], 1, num_symbols],tf.float32))#(batch * len) * 3 * num_symbols
            print "result_p_expand.shape:", result_p_expand.get_shape()
            result_symbol_1 = tf.nn.softmax(tf.matmul(local_outputs, weights1) + bias1)
            result_symbol_2 = tf.nn.softmax(tf.matmul(local_outputs, weights2) + bias2)
            result_symbol_3 = tf.nn.softmax(tf.matmul(local_outputs, weights3) + bias3)#(batch * len) * numsymbols
            result_symbol_1 = tf.expand_dims(result_symbol_1, axis=1)
            result_symbol_2 = tf.expand_dims(result_symbol_2, axis=1)
            result_symbol_3 = tf.expand_dims(result_symbol_3, axis=1)
            result_symbol = tf.concat([result_symbol_1, result_symbol_2, result_symbol_3], axis=1)#(batch*len) * 3 * num_symbols
            print "result_symbol.shape:", result_symbol.get_shape()

            #calculate the hybird distribution
            result_final = tf.clip_by_value(tf.reduce_sum(result_symbol * result_p_expand, axis=1), 1e-10, 1)
            result_final = tf.Print(result_final, ["result_final", result_final, "result_final_sum", tf.reduce_sum(result_final, axis=1)])
            print "result_final", result_final.get_shape()

            #calculate the loss for the final hybird distribution
            onehot_symbol_labels = tf.one_hot(tf.reshape(local_labels, [-1]), num_symbols)
            print "onehot_symbol_labels", onehot_symbol_labels.get_shape()
            local_loss = -tf.reduce_sum(onehot_symbol_labels * tf.log(result_final), axis=1)
            local_loss = local_loss * local_masks
            local_loss = tf.Print(local_loss, ["local loss", local_loss], summarize = 5)
            loss = tf.reduce_sum(local_loss)
            print "loss:", loss, "loss.shape:", loss.get_shape()

            #calculate the total loss
            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            return (tf.constant(0.8) * loss + loss2) / total_size, loss / total_size

    return output_fn, sampled_sequence_loss

