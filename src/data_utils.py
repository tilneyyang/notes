import os
import re
import numpy as np
import sys

UNK_TOKEN = '_UNK_TOKEN'
UNK_TOKEN_ID = 0
PAD_TOKEN = '_PAD_TOKEN'
PAD_TOKEN_ID = 1

IBO2_LABEL = {
    'B': 0,
    'I': 1,
    'O': 2
}


def _generate_ibo2_tag(token):
    token_len = len(token)
    assert token_len > 0
    if len(token) == 1:
        return [IBO2_LABEL['O']]
    tags = [IBO2_LABEL['B']]
    for i in xrange(token_len - 1):
        tags.append(IBO2_LABEL['I'])
    return tags


def create_vocabulary(path_to_data_file, vocab_dir, vocab_size=0):
    """
    create vocabulary from data file
    :param path_to_data_file:
    :param vocab_dir: dir to store output vocabulary
    :param vocab_size: store at most vocab_size most frequent chars
    :return:
        path to the vocab file
    """
    vocab_file = os.path.join(vocab_dir, 'vocab_%s' % vocab_size)
    if os.path.exists(vocab_file):
        sys.stderr.write('vocab file %s exists! we will use the exist one.\n' % vocab_file)
        return vocab_file
    unichars = {}
    with open(path_to_data_file) as iff:
        for line in iff:
            line = line.decode('utf8').strip()
            for char in line:
                if char not in unichars:
                    unichars[char] = 0
                unichars[char] += 1
    sorted_chars = sorted(unichars.items(), key=lambda x: x[1], reverse=True)
    if vocab_size > 0:
        sorted_chars = sorted_chars[:vocab_size]
    vocab = [UNK_TOKEN, PAD_TOKEN]
    for char, _ in sorted_chars:
        if len(char.strip()) > 0:
            vocab.append(char)
    vocab = dict(enumerate(vocab))
    with open(vocab_file, 'w') as vf:
        for item in vocab.items():
            vf.write('%s\t%s\n' % (item[0], item[1].encode('utf8')))
    return vocab_file


def read_vocabulary(path_to_vocab):
    """
    read vocabulary from a pre-generated vocab file
    :param path_to_vocab:
    :return:
    """
    if not os.path.exists(path_to_vocab):
        raise ValueError('vocab file %s not exists! please run create_vocabulary() first.' % path_to_vocab)
    with open(path_to_vocab) as vf:
        vocab = {}
        rev_vocab = {}
        for line in vf:
            line = line.decode('utf8').strip()
            chars_and_ids = re.split(ur'\s', line)
            assert len(chars_and_ids) == 2 or len(chars_and_ids) == 0
            vocab[chars_and_ids[1]] = int(chars_and_ids[0])
            rev_vocab[int(chars_and_ids[0])] = chars_and_ids[1]
        return vocab, rev_vocab


def sentence_to_token_ids(vocab, token):
    return [vocab[c] if c in vocab else UNK_TOKEN_ID for c in token]


def read_data(path_to_data_file, path_to_vocabulary, strip_chars='',  maximum_size=0, encoding_func=_generate_ibo2_tag):
    """
    read training or testing data, and generate tag for them. The data should contain one sentence per line,
    and tokens of the sentence is separated by space or tab (ur'\s+').
    :param path_to_data_file:
    :param encoding_func:function to generate tags
    :param strip_chars: characters will be stripped from start and end of the sentence
    :return:
        a list contains of all unicode encoding sentences characters to their tags
    """
    data = []
    count = 0
    vocab, _ = read_vocabulary(path_to_vocabulary)

    with open(path_to_data_file) as ipf:
        for line in ipf:
            line = line.strip().decode('utf8')
            if len(line) == 0:
                continue
            count += 1
            if strip_chars is not None and len(strip_chars) > 0:
                if not isinstance(strip_chars, unicode):
                    strip_chars = strip_chars.decode('utf8')
                line = line.strip(strip_chars)
            tokens = re.split(ur'\s+', line)
            all_tags = []
            all_token_chars = []
            for token in tokens:
                all_token_chars.extend(sentence_to_token_ids(vocab, token))
                tags = encoding_func(token)
                all_tags.extend(tags)
            # we have one tags per token char
            assert len(all_tags) == len(all_token_chars)
            data.append([all_token_chars, all_tags])
            if 0 < maximum_size < count:
                break
    return data


def data_iterator(raw_data, batch_size, num_steps, encoding_func=_generate_ibo2_tag):
    """Iterate on the raw data.

    This generates batch_size pointers into the raw data, and allows
    minibatch iteration along these pointers.

    Args:
      raw_data: one of the raw data outputs from raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
;
    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.

    Raises:
      ValueError: if batch_size or num_steps are too high.
    """
    batch_x = np.zeros([batch_size, num_steps], dtype=np.int32)
    batch_y = np.zeros([batch_size, num_steps], dtype=np.int32)
    for i in xrange(len(raw_data)):
        if i != 0 and i % batch_size == 0:
            yield batch_x, batch_y
        data = raw_data[i]
        x = [j for j in data[0]]
        y = [j for j in data[1]]
        assert len(x) == len(y)
        # pad the sentence
        if len(x) < num_steps:
            for j in xrange(num_steps - len(x)):
                x.append(PAD_TOKEN_ID)
                y.extend(encoding_func([PAD_TOKEN]))
        try:
            batch_x[i % batch_size] = x[:num_steps]
            batch_y[i % batch_size] = y[:num_steps]
        except Exception, e:
            print ''

if __name__ == '__main__':
    vocab_path = create_vocabulary('test', './')
    data = read_data('test', vocab_path)
    for i, j in read_vocabulary(vocab_path)[0].items():
        print i, j
    for i, j in read_vocabulary(vocab_path)[1].items():
        print i, j
    for x, y in data_iterator(data, 2, 5):
        print x, y
