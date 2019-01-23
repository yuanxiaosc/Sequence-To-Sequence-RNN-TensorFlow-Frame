import tensorflow as tf
from english2french_NMT_TensorFlow.data_process import load_preprocess, load_params, sentence_to_seq

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = load_preprocess()
load_path = load_params()


translate_sentence = 'he saw a old yellow truck .'

translate_sentence_seq = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence_seq],
                                         target_sequence_length: [len(translate_sentence_seq)],
                                         keep_prob: 1.0})

print("translate_logits:\t",translate_logits)
print('\n')

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence_seq]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence_seq]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in translate_logits[0]]))
print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits[0]])))

