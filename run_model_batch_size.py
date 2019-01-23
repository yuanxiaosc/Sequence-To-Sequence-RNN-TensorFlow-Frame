import tensorflow as tf

from english2french_NMT_TensorFlow.data_process import get_batches, load_preprocess, save_params
from english2french_NMT_TensorFlow.seq2seq_RNN_model_batch_size import Sequence_to_Sequence_Model, get_accuracy
display_step = 300
save_path = 'batch_size_checkpoints/dev'
epochs = 30
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

s2s_model = Sequence_to_Sequence_Model(
                                       batch_size=batch_size,
                                       source_vocab_size=source_vocab_size,
                                       target_vocab_size=target_vocab_size)

(train_op, cost),inference_logits, (input_data, targets, lr, target_sequence_length, keep_prob) = s2s_model.produce_model()


print('\n')
print('\n')
print('\n')
print('start train model...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    number_of_print = 0
    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 keep_prob: keep_probability})

            if batch_i % display_step == 0 and batch_i > 0:
                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})

                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)
                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print(
                    'Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                    .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')

save_params(save_path)