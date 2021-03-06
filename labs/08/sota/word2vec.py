#!/usr/bin/env python3
import gensim
import logging
import numpy as np

import morpho_dataset


def extract_sentences(dataset):
    return dataset._factors[0].strings


class WikiSentences:
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        with open(self.fname, "r") as f:
            for sent in f:
                yield sent.split()


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_word_count", default=5, type=int)
    parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.datafile = "data/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value.replace('/','|')) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("data"): os.mkdir("data")

    #sentences = WikiSentences("data/wiki-cs-tokenized.txt")
    #model = gensim.models.Word2Vec(sentences, min_count=args.min_word_count, size=args.we_dim, workers=args.threads)
    #model.save(args.datafile)

    # Load the data
    train = morpho_dataset.MorphoDataset("czech-pdt-train.txt", shuffle_batches=False)
    # dev = morpho_dataset.MorphoDataset("czech-pdt-dev.txt", shuffle_batches=False, train=train, all_words=train._factors[train.FORMS])
    # test = morpho_dataset.MorphoDataset("czech-pdt-test.txt", shuffle_batches=False, train=train, all_words=dev._factors[dev.FORMS])

    args.datafile = "data/word2vec.py-2018-04-27_105543-mwc=5,t=1,wd=128"
    w2v_model = gensim.models.Word2Vec.load(args.datafile)
    w2v = np.random.random((len(train.factors[train.FORMS].words), args.we_dim))
    for idx, word in enumerate(train.factors[train.FORMS].words):
        if word in w2v_model:
            w2v[idx] = w2v_model[word]
    np.save(args.datafile + "-only-matching", w2v)
