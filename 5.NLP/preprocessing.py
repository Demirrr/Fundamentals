import re
from collections import defaultdict
from heapq import nlargest
from operator import itemgetter


def tokenize(lines):
    """
    Split text by space and puctuation into sentences
    :param lines: sequence of text containing line
    :return: yield tokenized text line by line
    """
    tokenizer = re.compile('[ ,.?!]')

    for line in lines:
        line = line.rstrip('\n')
        sentence = tokenizer.split(line)
        sentence = list(filter(None, sentence))
        if sentence:
            yield sentence


def lower(sentences):
    """
    Lower case sentences
    :param sentences: sequence of sentence (list of words)
    :return: yield sentence with lower cased words
    """
    for sentence in sentences:
        words = [word.lower() for word in sentence]
        yield words


def prepend_caret(sentences):
    """
    Prepend caret as marker for sentence beginning
    :param sentences: sequence of sentence (list of words)
    :return: yield sentence with EOS marker
    """
    for sentence in sentences:
        words = ['^'] + sentence
        yield words


def build_dictionary(sentences, size):
    """
    Create dictionary containing most frequent words in the sentences
    :param sentences: sequence of sentence that contains words
        Caution: the sequence might be exhausted after calling this function!
    :param size: size of dictionary you want
    :return: dictionary that maps word to index (starting from 1)
    """
    dictionary = defaultdict(int)
    for sentence in sentences:
        for token in sentence:
            dictionary[token] += 1
    frequent_pairs = nlargest(size, dictionary.items(), itemgetter(1))
    words, frequencies = zip(*frequent_pairs)
    result = {word: index + 1 for index, word in enumerate(words)}
    return result


def to_indices(sentences, dictionary):
    """
    Convert sentence to list of indices
    :param sentences: sequence of sentences
    :param dictionary: dictionary that maps word to index
    :return: yield list of indices (0 if not in dictionary)
    """
    for sentence in sentences:
        indices = []
        for word in sentence:
            if word in dictionary:
                indices.append(dictionary[word])
            else:
                indices.append(0)
        yield indices
