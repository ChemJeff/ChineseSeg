#coding = utf-8

'''
Load data given in .txt file , do some preprocessing 
and then save to .pkl files
'''

import pickle

def add_freq(uchar, freq_dict):
    if uchar not in freq_dict.keys():
        freq_dict[uchar] = 1
    else:
        freq_dict[uchar] += 1

def build_vocab_list(file_path, encoding='utf-8', out_file=None):

    vocab_list = set()
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            for word in line.split():
                vocab_list.add(word)
    if out_file is not None:
        with open(out_file, 'w', encoding=encoding) as f:
            f.write(" ".join(list(vocab_list)))

    return list(vocab_list)



def build_vocab_tag(file_path, encoding='utf-8'):

    char_freq = dict()
    word2id = {}
    id2word = {}
    tag2id = {'S': 0, 'B': 1, 'M': 2, 'E': 3}
    id2tag = {}

    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            # 注意BOM开头字节的问题(window notepad)
            words = line.split() 
            charline = [uchar for uchar in "".join(words)]
            for uchar in charline:
                add_freq(uchar, char_freq)

    sorted_freq = sorted(char_freq.items(), key=lambda x:x[1], reverse=True)
    sorted_freq.append(('[UNK]', 0))

    for idx, item in enumerate(sorted_freq):
        word2id[item[0]] = idx
        id2word[idx] = item[0]

    assert len(word2id) == len(id2word)

    for i in tag2id.items():
        id2tag[i[1]] = i[0]

    print("Build vocab size = %d, including [UNK]" % (len(word2id)))
    return word2id, id2word, tag2id, id2tag

def corpus_convert(file_path, split, encoding='utf-8'):

    assert split in ['train', 'test']

    print("Data split: %s" % (split))
    eline_cnt = 0 # count of emptylines

    corpus = dict()
    corpus['split'] = split
    corpus['len'] = 0
    corpus['words'] = []
    if split == 'train':
        corpus['tags'] = []

    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            words = line.split()
            charline = [uchar for uchar in "".join(words)]

            if split == 'test':
                corpus['words'].append(charline)
                corpus['len'] += 1
                continue

            charlinel = len(charline)
            tagline = []
            for word in words:
                lenw = len(word)
                if lenw == 1:
                    tagline += ['S']
                else:
                    tagline += ['B'] + ['M'] * (lenw - 2) + ['E']
            assert charlinel == len(tagline)
            if charlinel == 0:
                eline_cnt += 1 #emptyline
                continue
            corpus['words'].append(charline)
            corpus['tags'].append(tagline)
            corpus['len'] += 1

    print("Num of sentences: %d" % (corpus['len']))
    print("Empty lines: %d" % (eline_cnt))

    return corpus


if __name__ == "__main__":
    # train_path = './data/train.txt'
    test_path = './data/test.txt'
    # word2id, id2word, tag2id, id2tag = bulid_vocab_tag(train_path)
    # with open('vocab_tag.pkl', 'wb') as f:
    #     pickle.dump((word2id, id2word, tag2id, id2tag), file=f)
    # with open('vocab_tag.pkl', 'rb') as f :
    #     w2i, i2w, t2i, i2t = pickle.load(f)
    #     print(w2i, i2w, t2i, i2t, sep='\n'
    # train_corpus = corpus_convert(train_path, 'train')
    test_corpus = corpus_convert(test_path, 'test')
    # with open('train_corpus.pkl', 'wb') as f:
    #     pickle.dump(train_corpus, file=f)
    with open('test_corpus.pkl', 'wb') as f:
        pickle.dump(test_corpus, file=f)
    # print(train_corpus)
    print(test_corpus)