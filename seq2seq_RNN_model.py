from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


class Sequence_to_Sequence_Model(object):

    def __init__(self, source_vocab_size=232, target_vocab_size=358):
        self.rnn_size = 128
        self.num_layers = 3

        self.encoding_embedding_size = 200
        self.decoding_embedding_size = 200

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        # Custom Special Coding
        self.SPECIAL_CHARACTER_ENCODING = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3}

    def enc_dec_model_inputs(self,):
        inputs = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')

        target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
        max_target_len = tf.reduce_max(target_sequence_length)

        return inputs, targets, target_sequence_length, max_target_len

    def hyperparam_inputs(self,):
        lr_rate = tf.placeholder(tf.float32, name='lr_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        return lr_rate, keep_prob

    def process_decoder_input(self, target_data):
        """
        just for delect last row('<EOS>'->end_id) and add a new first row('<GO>'->start_id)
        Preprocess target data for encoding
        :return: Preprocessed target data
        """
        # get '<GO>' id
        go_id = self.SPECIAL_CHARACTER_ENCODING['<GO>']

        #after_slice = tf.strided_slice(target_data, [0, 0], [self.batch_size, -1], [1, 1])
        #after_concat = tf.concat([tf.fill([self.batch_size, 1], go_id), after_slice], 1)

        #remove last column
        target_first_column = target_data[:, :-1]
        #get first column
        target_after_slice = target_data[:, 1:]
        #producte first first column
        target_first_ones_column = tf.ones_like(target_first_column)
        #profucte go_id first column
        target_go_id_first_column = target_first_ones_column + (go_id - 1)

        #concat go_id first column and remaining columns
        after_concat = tf.concat([target_go_id_first_column, target_after_slice], 1)

        return after_concat


    def encoding_layer(self, rnn_inputs, keep_prob):
        """
        :return: tuple (RNN output, RNN state)
        """
        embed = tf.contrib.layers.embed_sequence(rnn_inputs,
                                                vocab_size=self.source_vocab_size,
                                                 embed_dim=self.encoding_embedding_size)

        stacked_cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(self.rnn_size), keep_prob) for _ in range(self.num_layers)])

        outputs, state = tf.nn.dynamic_rnn(stacked_cells,
                                           embed,
                                           dtype=tf.float32)
        return outputs, state

    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input,
                             target_sequence_length, output_layer, max_target_sequence_length, keep_prob):
        """
        Create a training process in decoding layer
        :return: BasicDecoderOutput containing training logits and sample_id
        """
        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                                 output_keep_prob=keep_prob)

        # for only input layer
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,
                                                   target_sequence_length)

        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                  helper,
                                                  encoder_state,
                                                  output_layer)

        # unrolling the decoder layer
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                          impute_finished=True,
                                                          maximum_iterations=max_target_sequence_length)
        return outputs

    def decoding_layer_infer(self, encoder_state, dec_cell, dec_embeddings,output_layer, max_target_sequence_length,keep_prob):
        """
        Create a inference process in decoding layer
        :return: BasicDecoderOutput containing inference logits and sample_id
        """
        start_of_sequence_id = self.SPECIAL_CHARACTER_ENCODING['<GO>']
        end_of_sequence_id = self.SPECIAL_CHARACTER_ENCODING['<EOS>']

        def _produce_a_start_id_vector(encoder_state, start_of_sequence_id):
            """just for produce a vector shape: (batch_size,) value: start id"""
            #shape:(batch_size, encoder_state_columns)  type: float32
            ones_like_encoder_state = tf.ones_like(encoder_state[0].c)
            #shape:(batch_size, 1) values: 1            type: float32
            ones_like_encoder_state_a_column = ones_like_encoder_state[:,:1]
            #shape:(batch_size, 1) values: start_token id  type: float32
            start_of_sequence_id_a_column = ones_like_encoder_state_a_column + (start_of_sequence_id - 1)
            start_of_sequence_id_a_column = tf.cast(start_of_sequence_id_a_column, tf.int32)
            ##shape:(batch_size,) start_tokens must be a vector  values: start_token id type: int32
            start_of_sequence_id_vector = tf.reshape(start_of_sequence_id_a_column, [-1], name="start_tokens")
            return start_of_sequence_id_vector

        start_of_sequence_id_vector = _produce_a_start_id_vector(encoder_state, start_of_sequence_id)

        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                                 output_keep_prob=keep_prob)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                          start_of_sequence_id_vector,
                                                          end_of_sequence_id)

        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                  helper,
                                                  encoder_state,
                                                  output_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                          impute_finished=True,
                                                          maximum_iterations=max_target_sequence_length)
        return outputs


    def decoding_layer(self, dec_input, encoder_state, target_sequence_length, max_target_sequence_length, keep_prob):
        """
        Create decoding layer
        :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
        """
        dec_embeddings = tf.Variable(tf.random_uniform([self.target_vocab_size, self.decoding_embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.rnn_size) for _ in range(self.num_layers)])

        with tf.variable_scope("decode"):
            output_layer = tf.layers.Dense(self.target_vocab_size)
            train_output = self.decoding_layer_train(encoder_state,
                                                cells,
                                                dec_embed_input,
                                                target_sequence_length,
                                                output_layer,
                                                max_target_sequence_length,
                                                     keep_prob)

        with tf.variable_scope("decode", reuse=True):
            infer_output = self.decoding_layer_infer(encoder_state,
                                                cells,
                                                dec_embeddings,
                                                output_layer,
                                                max_target_sequence_length,
                                                     keep_prob)

        return (train_output, infer_output)


    def produce_model(self,):
        input_data, targets, target_sequence_length, max_target_sequence_length = self.enc_dec_model_inputs()

        lr, keep_prob = self.hyperparam_inputs()

        input_data = tf.reverse(input_data, [-1])
        enc_outputs, enc_states = self.encoding_layer(input_data, keep_prob)

        dec_input = self.process_decoder_input(targets)

        train_output, infer_output = self.decoding_layer(dec_input, enc_states, target_sequence_length,
                                                         max_target_sequence_length, keep_prob)
        training_logits = tf.identity(train_output.rnn_output, name='logits')
        inference_logits = tf.identity(infer_output.sample_id, name='predictions')
        # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
        # - Returns a mask tensor representing the first N positions of each cell.
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function - weighted softmax cross entropy
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

        return (train_op, cost),inference_logits, (input_data, targets, lr, target_sequence_length, keep_prob)



def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0, 0), (0, max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0, 0), (0, max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))
