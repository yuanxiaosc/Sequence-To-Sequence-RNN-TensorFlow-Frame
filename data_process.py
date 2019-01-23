import os
import pickle
import copy
import numpy as np
from collections import Counter


#Original file path
source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'

#Custom Special Coding
SPECIAL_CHARACTER_ENCODING = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

def show_sample_statistics():
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    print('Dataset Brief Stats')
    print('* number of unique words in English sample sentences: {}\
            [this is roughly measured/without any preprocessing]'.format(len(Counter(source_text.split()))))
    print('* number of unique words in French sample sentences: {}\
            [this is roughly measured/without any preprocessing]'.format(len(Counter(target_text.split()))))
    print()

    english_sentences = source_text.split('\n')
    print('* English sentences')
    print('\t- number of sentences: {}'.format(len(english_sentences)))
    print('\t- avg. number of words in a sentence: {}'.format(np.average([len(sentence.split()) for sentence in english_sentences])))

    french_sentences = target_text.split('\n')
    print('* French sentences')
    print('\t- number of sentences: {} [data integrity check / should have the same number]'.format(len(french_sentences)))
    print('\t- avg. number of words in a sentence: {}'.format(np.average([len(sentence.split()) for sentence in french_sentences])))
    print()

    sample_sentence_range = (0, 5)
    side_by_side_sentences = list(zip(english_sentences, french_sentences))[sample_sentence_range[0]:sample_sentence_range[1]]
    print('* Sample sentences range from {} to {}'.format(sample_sentence_range[0], sample_sentence_range[1]))

    for index, sentence in enumerate(side_by_side_sentences):
        en_sent, fr_sent = sentence
        print('[{}-th] sentence'.format(index+1))
        print('\tEN: {}'.format(en_sent))
        print('\tFR: {}'.format(fr_sent))
        print()

def create_lookup_tables(text, SPECIAL_CHARACTER_ENCODING):
    # make a list of unique words
    vocab = set(text.split())

    # (1)
    # starts with the special tokens
    vocab_to_int = copy.copy(SPECIAL_CHARACTER_ENCODING)

    # the index (v_i) will starts from 4 (the 2nd arg in enumerate() specifies the starting index)
    # since vocab_to_int already contains special tokens
    for v_i, v in enumerate(vocab, len(SPECIAL_CHARACTER_ENCODING)):
        vocab_to_int[v] = v_i

    # (2)
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
        1st, 2nd args: raw string text to be converted
        3rd, 4th args: lookup tables for 1st and 2nd args respectively

        return: A tuple of lists (source_id_text, target_id_text) converted
    """
    # empty list of converted sentences
    source_text_id = []
    target_text_id = []

    # make a list of sentences (extraction)
    source_sentences = source_text.split("\n")
    target_sentences = target_text.split("\n")

    #max_source_sentence_length = max([len(sentence.split(" ")) for sentence in source_sentences])
    #max_target_sentence_length = max([len(sentence.split(" ")) for sentence in target_sentences])

    # iterating through each sentences (# of sentences in source&target is the same)
    for i in range(len(source_sentences)):
        # extract sentences one by one
        source_sentence = source_sentences[i]
        target_sentence = target_sentences[i]

        # make a list of tokens/words (extraction) from the chosen sentence
        source_tokens = source_sentence.split(" ")
        target_tokens = target_sentence.split(" ")

        # empty list of converted words to index in the chosen sentence
        source_token_id = []
        target_token_id = []

        for index, token in enumerate(source_tokens):
            if (token != ""):
                source_token_id.append(source_vocab_to_int[token])

        for index, token in enumerate(target_tokens):
            if (token != ""):
                target_token_id.append(target_vocab_to_int[token])

        # put <EOS> token at the end of the chosen target sentence
        # this token suggests when to stop creating a sequence
        target_token_id.append(target_vocab_to_int['<EOS>'])

        # add each converted sentences in the final list
        source_text_id.append(source_token_id)
        target_text_id.append(target_token_id)

    return source_text_id, target_text_id


def preprocess_and_save_data(source_path, target_path, text_to_ids, SPECIAL_CHARACTER_ENCODING=SPECIAL_CHARACTER_ENCODING):
    # Preprocess

    # load original data (English, French)
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    # to the lower case
    source_text = source_text.lower()
    target_text = target_text.lower()

    # create lookup tables for English and French data
    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text,SPECIAL_CHARACTER_ENCODING)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text,SPECIAL_CHARACTER_ENCODING)
    print("~" * 100)
    print("source vocab word numbers:\t",len(source_vocab_to_int))
    print("target vocab word numbers:\t",len(target_vocab_to_int))
    print("~"*100)
    # create list of sentences whose words are represented in index
    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save data for later use
    pickle.dump((
        (source_text, target_text),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab)), open('preprocess.p', 'wb'))
    print("save all prepared data to prerocess.p")

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


def sentence_to_seq(sentence, vocab_to_int):
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])

    return results


def load_preprocess():
    with open('preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


def save_params(params):
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)



if __name__=="__main__":
    show_sample_statistics()
    preprocess_and_save_data(source_path, target_path, text_to_ids, SPECIAL_CHARACTER_ENCODING=SPECIAL_CHARACTER_ENCODING)
