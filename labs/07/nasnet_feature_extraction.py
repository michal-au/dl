#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

import nets.nasnet.nasnet


class Dataset:
    def __init__(self, filename):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None

        self._permutation = np.arange(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            return True
        return False


class Network:
    WIDTH, HEIGHT = 224, 224

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Create NASNet
            images = 2 * (tf.tile(tf.image.convert_image_dtype(self.images, tf.float32), [1, 1, 1, 3]) - 0.5)
            with tf.contrib.slim.arg_scope(nets.nasnet.nasnet.nasnet_mobile_arg_scope()):
                self.features, _ = nets.nasnet.nasnet.build_nasnet_mobile(images, num_classes=None, is_training=False)
            self.nasnet_saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

            # Load NASNet
            self.nasnet_saver.restore(self.session, args.nasnet)

    def extract_features(self, images):
        return self.session.run(self.features, {self.images: images})


if __name__ == "__main__":
    import argparse
    import os

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--nasnet", default="nets/nasnet/model.ckpt", type=str, help="NASNet checkpoint path.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    train = Dataset("nsketch-train.npz")
    dev = Dataset("nsketch-dev.npz")
    test = Dataset("nsketch-test.npz")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    if not os.path.exists("features"): os.mkdir("features")

    for dataset, name in [(train, "train"), (dev, "dev"), (test, "test")]:
        features = np.empty([0, 1056])
        all_labels = np.empty(0)
        i = 0
        while not dataset.epoch_finished():
            print(i)
            images, labels = dataset.next_batch(args.batch_size)
            features = np.vstack([features, network.extract_features(images)])
            all_labels = np.hstack([all_labels, labels])
            i += 1

        np.savez(open("features/nasnet-{}.npz".format(name), "wb"), images=features, labels=all_labels)
