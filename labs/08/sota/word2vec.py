#!/usr/bin/env python3
import gensim
import logging
import numpy as np

import morpho_dataset


def extract_sentences(dataset):
    return dataset._factors[0].strings


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
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("data"): os.mkdir("data")

    # Load the data
    train = morpho_dataset.MorphoDataset("czech-pdt-train.txt", shuffle_batches=False)
    dev = morpho_dataset.MorphoDataset("czech-pdt-dev.txt", shuffle_batches=False)
    test = morpho_dataset.MorphoDataset("czech-pdt-test.txt", shuffle_batches=False)
    # sentences = extract_sentences(dev)
    # sentences.extend(extract_sentences(dev))
    # sentences.extend(extract_sentences(test))
    #
    # model = gensim.models.Word2Vec(sentences, min_count=args.min_word_count, size=args.we_dim, workers=args.threads)
    # model.save(args.datafile)

    w2v_model = gensim.models.Word2Vec.load("data/word2vec.py-2018-04-24_195427-mwc=5,t=2,wd=128")
    print(w2v_model.wv.vector_size)
    for word in train.factors[train.FORMS].words:
        print(word, w2v_model[word] if word in w2v_model else )

