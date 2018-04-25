#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import gensim

import morpho_dataset


class MorphoAnalyzer:
    """ Loader for data of morphological analyzer.

    The loaded analyzer provides an only method `get(word)` returning
    a list of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    class LemmaTag:
        def __init__(self, lemma, tag):
            self.lemma = lemma
            self.tag = tag

    def __init__(self, filename):
        self.analyses = {}

        with open(filename, "r", encoding="utf-8") as analyzer_file:
            for line in analyzer_file:
                line = line.rstrip("\n")
                columns = line.split("\t")

                analyses = []
                for i in range(1, len(columns) - 1, 2):
                    analyses.append(MorphoAnalyzer.LemmaTag(columns[i], columns[i + 1]))
                self.analyses[columns[0]] = analyses

    def get(self, word):
        return self.analyses.get(word, [])


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_chars, num_tags):
        def construct_layer(config_line, input_layer):
            layer_type, *params = config_line.split("-")
            if layer_type == "R":
                return tf.layers.dense(input_layer, int(params[0]), activation=tf.nn.relu)
            elif layer_type == "D":
                return tf.layers.dropout(input_layer, float(params[0]), training=self.is_training)

        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")

            self.tags_mask = tf.placeholder(tf.float32, [None, num_tags], name="tags_mask")

            cell_constructor = tf.nn.rnn_cell.BasicLSTMCell if args.rnn_cell == "LSTM" else tf.nn.rnn_cell.GRUCell
            fw_cell = cell_constructor(args.rnn_cell_dim)
            bw_cell = cell_constructor(args.rnn_cell_dim)
            word_embeddings = tf.get_variable(name="word_embeddings", shape=[num_words, args.we_dim])
            words_embedded = tf.nn.embedding_lookup(word_embeddings, self.word_ids, name="embedding_lookup")

            if args.cle_dim:
                char_embeddings = tf.get_variable(name="char_embeddings", shape=[num_chars, args.cle_dim])
                charseqs_embedded = tf.nn.embedding_lookup(char_embeddings, self.charseqs, name="char_embedding_lookup")
                kernel_specific_embs = []
                for ks in range(2, args.cnne_max + 1):
                    convoluted_seq = tf.layers.conv1d(charseqs_embedded, args.cnne_filters, ks, strides=1, padding="valid")
                    kernel_specific_embs.append(
                        tf.layers.max_pooling1d(convoluted_seq, 1000, strides=1, padding="same")[:,1,:]
                    )
                cnn_embeddings = tf.concat(kernel_specific_embs, axis=1)
                words_cnn_embedded = tf.nn.embedding_lookup(cnn_embeddings, self.charseq_ids)
                words_embedded = tf.concat([words_embedded, words_cnn_embedded], axis=2)

            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, words_embedded, sequence_length=self.sentence_lens, dtype=tf.float32)
            rnn_output = tf.concat(rnn_outputs, axis=2, name="concat_rnn_outputs")

            if args.rnn_output_process_architecture:
                hidden_layer = rnn_output
                for layer_config_line in args.rnn_output_process_architecture.split(","):
                    hidden_layer = construct_layer(layer_config_line, hidden_layer)
                rnn_output = hidden_layer

            output_layer = tf.layers.dense(rnn_output, num_tags, activation=None)

            self.predictions = tf.argmax(output_layer, axis=2)
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            if args.analyzer:
                # tag_mask_tensor = tf.get_variable("tag_mask", shape=tag_mask.shape)  # tag_mask
                # tag_mask_tensor = tf.get_variable("tag_mask", shape=tag_mask.shape, trainable=False)
                # tag_mask_tensor = tf.assign(tag_mask_tensor, tag_mask)
                words_tags_masked = tf.nn.embedding_lookup(self.tags_mask, self.word_ids, name="tag_mask_lookup")
                output_layer = tf.multiply(output_layer, words_tags_masked, name="masking")

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.tags, output_layer, weights=weights)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size, tags_masks):
        while not train.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size, including_charseqs=True)
            batch_tags_masks = tags_masks[word_ids, :]
            print(batch_tags_masks)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS],
                              self.tags_mask: batch_tags_masks
                              })

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        tags = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            tags.extend(self.session.run(self.predictions,
                                         {self.sentence_lens: sentence_lens,
                                          self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                                          self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS]}))
        return tags


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
    parser.add_argument("--max_sents", default=None, type=int, help="Nb of senteces used for training")
    parser.add_argument("--rnn_output_process_architecture", default=None, type=str)

    parser.add_argument("--cle_dim", default=0, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--cnne_filters", default=16, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnne_max", default=4, type=int, help="Maximum CNN filter length.")

    parser.add_argument("--pretrained_w2v", default=None, type=str, help="Path to pretrained gensim w2v model.")

    parser.add_argument("--analyzer", default=False, type=bool, help="Should the morpho analyzer be used?")

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = morpho_dataset.MorphoDataset("czech-pdt-train.txt", max_sentences=args.max_sents)
    dev = morpho_dataset.MorphoDataset("czech-pdt-dev.txt", train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset("czech-pdt-test.txt", train=train, shuffle_batches=False)

    tag_mask = None
    if args.analyzer:
        try:
            tag_mask = np.load("data/tag_mask.npy")
        except FileNotFoundError:
            analyzer_dictionary = MorphoAnalyzer("czech-pdt-analysis-dictionary.txt")
            analyzer_guesser = MorphoAnalyzer("czech-pdt-analysis-guesser.txt")
            height, width = len(train.factors[train.FORMS].words), len(train.factors[train.TAGS].words)
            tag_mask = np.zeros([height, width], dtype=np.bool)

            row_idcs, column_idcs = [], []
            for idx, w in enumerate(train.factors[train.FORMS].words):
                tags = analyzer_dictionary.get(w) or analyzer_guesser.get(w)
                tag_ids = [train.factors[train.TAGS].words_map.get(lt.tag) for lt in tags if train.factors[train.TAGS].words_map.get(lt.tag)]
                if not tag_ids:
                    row_idcs.extend([idx] * width)
                    column_idcs.extend(range(width))
                else:
                    row_idcs.extend([idx] * len(tag_ids))
                    column_idcs.extend(tag_ids)

            tag_mask[(row_idcs, column_idcs)] = True
            tag_mask = tag_mask.astype(np.float32)
            np.save("data/tag_mask", tag_mask)

    # TODO: vlastni embeddingy jako inicializace
    # if args.pretrained_w2v:
    #     w2v_model = gensim.models.Word2Vec.load(args.pretrained_w2v)
    #     for word_form in enumerate(train.factors[train.FORMS].words):
    #         w2v_model[word_form]


    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      len(train.factors[train.TAGS].words))

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size, tag_mask)
        exit()
        network.evaluate("dev", dev, args.batch_size)

    # Predict test data
    with open("{}/tagger_sota_test.txt".format(args.logdir), "w") as test_file:
        forms = test.factors[test.FORMS].strings
        tags = network.predict(test, args.batch_size)
        for s in range(len(forms)):
            for i in range(len(forms[s])):
                print("{}\t_\t{}".format(forms[s][i], test.factors[test.TAGS].words[tags[s][i]]), file=test_file)
            print("", file=test_file)
