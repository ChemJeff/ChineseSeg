import pickle
import utils.dataPreprocess as dataPreprocess

if __name__ == "__main__":
    data_path = "./data/"
    word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = dataPreprocess.build_vocab_tag(data_path + 'train.txt')
    with open(data_path + 'vocab_tag.pkl', 'wb') as f:
        pickle.dump((word_to_ix, ix_to_word, tag_to_ix, ix_to_tag), file=f)

    train_corpus = dataPreprocess.corpus_convert(data_path + 'train.txt', 'train')
    with open(data_path + 'train_corpus.pkl', 'wb') as f:
        pickle.dump(train_corpus, file=f)

    test_corpus = dataPreprocess.corpus_convert(data_path + 'test.txt', 'test')
    with open(data_path + 'test_corpus.pkl', 'wb') as f:
        pickle.dump(test_corpus, file=f)