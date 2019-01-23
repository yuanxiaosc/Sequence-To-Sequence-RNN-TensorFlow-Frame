import tensorflow as tf
import numpy as np
from english2french_NMT_TensorFlow.seq2seq_RNN_model import Sequence_to_Sequence_Model, get_accuracy
from english2french_NMT_TensorFlow.data_process import get_batches, load_preprocess, save_params

def get_accuracy_V2(target, logits):
    """
    Calculate accuracy
    """
    if target.shape[1]==logits.shape[1]:
        return np.mean(np.equal(target, logits))
    if target.shape[1]>logits.shape[1]:
        logits = np.pad(
            logits,
            [(0, 0), (0, target.shape[1] - logits.shape[1])],
            'constant')
    else:
        target = np.pad(
            target,
            [(0, 0), (0, logits.shape[1] - target.shape[1])],
            'constant')
    return np.mean(np.equal(target, logits))

display_step = 300
save_path = 'checkpoints/dev'
epochs = 15
batch_size = 128
learning_rate = 0.001
keep_probability = 0.5


(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = load_preprocess()

print(len(source_int_text))
print(len(target_int_text))
print(source_vocab_to_int)
print(target_vocab_to_int)

print('^'*50)
# Split data to training and validation sets
#train_source = source_int_text[:int(len(source_int_text)*0.8)]
#train_target = target_int_text[:int(len(target_int_text)*0.8)]
#valid_source = source_int_text[int(len(source_int_text)*0.8):]
#valid_target = target_int_text[int(len(target_int_text)*0.8):]
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
print(len(train_source))
print(len(train_target))
print(len(valid_source))
print(len(valid_target))

#get valid data
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths )\
    = next(get_batches(  valid_source,
                         valid_target,
                         batch_size,
                         source_vocab_to_int['<PAD>'],
                         target_vocab_to_int['<PAD>']))

#create model
source_vocab_size=len(source_vocab_to_int)
target_vocab_size=len(target_vocab_to_int)
from english2french_NMT_TensorFlow.seq2seq_RNN_model import Sequence_to_Sequence_Model

s2s_model = Sequence_to_Sequence_Model(source_vocab_size=source_vocab_size,
                                       target_vocab_size=target_vocab_size)

input_data, targets, target_sequence_length, max_target_sequence_length = s2s_model.enc_dec_model_inputs()

lr, keep_prob = s2s_model.hyperparam_inputs()

input_data = tf.reverse(input_data, [-1])
dec_input = s2s_model.process_decoder_input(targets)
enc_outputs, enc_states = s2s_model.encoding_layer(input_data, keep_prob)
train_output, infer_output = s2s_model.decoding_layer(dec_input, enc_states, target_sequence_length,
                                                 max_target_sequence_length, keep_prob)

training_logits = tf.identity(train_output.rnn_output, name='logits')
inference_logits = tf.identity(infer_output.sample_id, name='predictions')


print('start train model...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    number_of_print = 0
    for epoch_i in range(1):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_sources_batch,
                 target_sequence_length: valid_targets_lengths,
                 keep_prob: 1.0})
            print(batch_valid_logits.shape)
            print(batch_valid_logits[0])
            print(valid_targets_batch.shape)
            print(valid_targets_batch[0])
            valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)
            print(valid_acc)
            break
