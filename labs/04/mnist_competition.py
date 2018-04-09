#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

#i bs=100, c=CB-20-3-2-same,M-3-2,F,D-0.5,R-100, e=100, t=1


class Network:
    WIDTH = 28
    HEIGHT = 28
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
                return tf.layers.conv2d(input_layer, int(params[0]), int(params[1]), int(params[2]), params[3],
                                        activation=tf.nn.relu)
            if layer_type == "CB":
                cnn = tf.layers.conv2d(input_layer, int(params[0]), int(params[1]), int(params[2]), params[3], activation=None, use_bias=False)
                bn = tf.layers.batch_normalization(cnn, training=self.is_training)
                return tf.nn.relu(bn)
            elif layer_type == "M":
                return tf.layers.max_pooling2d(input_layer, int(params[0]), int(params[1]))
            elif layer_type == "F":
                return tf.layers.flatten(input_layer)
            elif layer_type == "R":
                return tf.layers.dense(input_layer, int(params[0]), activation=tf.nn.relu)
            elif layer_type == "D":
                return tf.layers.dropout(hidden_layer, float(params[0]), training=self.is_training)

        with self.session.graph.as_default():
            # TODO: Construct the network and training operation.
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            hidden_layer = self.images
            for layer_string in args.cnn.split(","):
                hidden_layer = construct_layer(layer_string, hidden_layer)
            output_layer = tf.layers.dense(hidden_layer, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)

            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
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

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, images, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.images: images, self.labels: labels, self.is_training: False})
        return accuracy

    def predict(self, images):
        return self.session.run([self.predictions], {self.images: images, self.is_training: False})[0]

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
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--cnn", default=None, type=str, help="Description of the CNN architecture.")
    args = parser.parse_args()

    # Create logdir name
    experiment_name = "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    args.logdir = "logs/" + experiment_name
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42,
                                            source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        network.evaluate("dev", mnist.validation.images, mnist.validation.labels)

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    test_labels = network.predict(mnist.test.images)
    with open("mnist_competition_test_{}.txt".format(experiment_name), "w") as test_file:
        for label in test_labels:
            print(label, file=test_file)
