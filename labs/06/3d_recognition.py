#!/usr/bin/env python3

"""
python 3d_recognition.py --epochs 120 --batch_size 128 --train_split 0.85 --architecture CB-16-3-1-sam
e,M-3-3,CB-16-3-1-same,M-2-2,F,D-0.6,R-50 --modelnet_dim 32
"""

import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._voxels = data["voxels"].astype(np.float32)
        self._labels = data["labels"] if "labels" in data else None

        self._shuffle_batches = shuffle_batches
        self._new_permutation()

    def _new_permutation(self):
        if self._shuffle_batches:
            self._permutation = np.random.permutation(len(self._voxels))
        else:
            self._permutation = np.arange(len(self._voxels))

    def split(self, ratio):
        split = int(len(self._voxels) * ratio)

        first, second = Dataset.__new__(Dataset), Dataset.__new__(Dataset)
        first._voxels, second._voxels = self._voxels[:split], self._voxels[split:]
        if self._labels is not None:
            first._labels, second._labels = self._labels[:split], self._labels[split:]
        else:
            first._labels, second._labels = None, None

        for dataset in [first, second]:
            dataset._shuffle_batches = self._shuffle_batches
            dataset._new_permutation()

        return first, second

    @property
    def voxels(self):
        return self._voxels

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._voxels[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._new_permutation()
            return True
        return False


class Network:
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):

        def construct_layer(config_line, input_layer):
            layer_type, *params = config_line.split("-")
            if layer_type == "C":
                return tf.layers.conv3d(input_layer, int(params[0]), int(params[1]), int(params[2]), params[3],
                                        activation=tf.nn.relu)
            if layer_type == "CB":
                cnn = tf.layers.conv3d(input_layer, int(params[0]), int(params[1]), int(params[2]), params[3], activation=None, use_bias=False)
                bn = tf.layers.batch_normalization(cnn, training=self.is_training)
                return tf.nn.relu(bn)
            elif layer_type == "M":
                return tf.layers.max_pooling3d(input_layer, int(params[0]), int(params[1]))
            elif layer_type == "F":
                return tf.layers.flatten(input_layer)
            elif layer_type == "R":
                return tf.layers.dense(input_layer, int(params[0]), activation=tf.nn.relu)
            elif layer_type == "D":
                return tf.layers.dropout(input_layer, float(params[0]), training=self.is_training)

        with self.session.graph.as_default():
            # Inputs
            self.voxels = tf.placeholder(
                tf.float32, [None, args.modelnet_dim, args.modelnet_dim, args.modelnet_dim, 1], name="voxels")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # NN architecture
            hidden_layer = self.voxels
            for layer_config_line in args.architecture.split(","):
                hidden_layer = construct_layer(layer_config_line, hidden_layer)

            output_layer = tf.layers.dense(hidden_layer, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer)

            # Training
            global_step = tf.train.create_global_step()
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(8):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, voxels, labels):
        self.session.run([self.training, self.summaries["train"]], {self.voxels: voxels, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, voxels, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.voxels: voxels, self.labels: labels, self.is_training: False})
        return accuracy

    def predict(self, voxels):
        return self.session.run(self.predictions, {self.voxels: voxels, self.is_training: False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--modelnet_dim", default=20, type=int, help="Dimension of ModelNet data.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train_split", default=None, type=float, help="Ratio of examples to use as train.")
    parser.add_argument("--architecture", default=None, type=str, help="nn architecture")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train, dev = Dataset("modelnet{}-train.npz".format(args.modelnet_dim)).split(args.train_split)
    test = Dataset("modelnet{}-test.npz".format(args.modelnet_dim), shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    acc_max = 0
    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            voxels, labels = train.next_batch(args.batch_size)
            network.train(voxels, labels)

        acc = network.evaluate("dev", dev.voxels, dev.labels)
        if acc > acc_max and acc > 0.96:
            print("new best acc: {}".format(acc))
            # Predict test data
            with open("{}/3d_recognition_testi-{}.txt".format(args.logdir, acc), "w") as test_file:
                while not test.epoch_finished():
                    voxels, _ = test.next_batch(args.batch_size)
                    labels = network.predict(voxels)

                    for label in labels:
                        print(label, file=test_file)
            acc_max = acc
