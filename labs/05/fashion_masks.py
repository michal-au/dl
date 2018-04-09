#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._masks = data["masks"] if "masks" in data else None

        self._permutation = np.random.permutation(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def masks(self):
        return self._masks

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm], self._masks[batch_perm]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images))
            return True
        return False


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
                return tf.layers.dropout(input_layer, float(params[0]), training=self.is_training)

        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.masks = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="masks")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Computation and training.
            hidden_layer = self.images
            for layer_string in args.encoder.split(","):
                hidden_layer = construct_layer(layer_string, hidden_layer)

            classification_layer = hidden_layer
            for layer_string in args.classification_decoder.split(","):
                classification_layer = construct_layer(layer_string, classification_layer)

            classification_layer = tf.layers.dense(classification_layer, self.LABELS, activation=None, name="output_layer")
            self.labels_predictions = tf.argmax(classification_layer, axis=1)

            mask_prediction_layer = hidden_layer
            for layer_string in args.mask_decoder.split(","):
                mask_prediction_layer = construct_layer(layer_string, mask_prediction_layer)

            available_labels = tf.cond(self.is_training, lambda: self.labels, lambda: self.labels_predictions)
            transp = tf.transpose(mask_prediction_layer, [0, 3, 1, 2])
            rang = tf.range(tf.shape(self.images)[0])
            idcs = tf.concat([tf.expand_dims(rang, 1), tf.expand_dims(tf.to_int32(available_labels), 1)], 1)

            self.masks_predictions_unrounded = tf.sigmoid(tf.expand_dims(tf.gather_nd(transp, idcs), axis=3, name="mask_pred_uround"))
            self.masks_predictions = tf.round(self.masks_predictions_unrounded, name="mask_pred_round")

            loss_a = tf.losses.sparse_softmax_cross_entropy(self.labels, classification_layer, scope="loss_a")

            a, b = tf.layers.flatten(self.masks), tf.layers.flatten(self.masks_predictions_unrounded)
            self.loss_b = tf.losses.sigmoid_cross_entropy(a, b)  # (1 - a) * tf.log(1 - b)
            # self.loss_b = tf.reduce_mean(a * tf.log(b) + (1 - a) * tf.log(1 - b))
            # intersection = tf.reduce_sum(self.masks_predictions * self.masks, axis=[1,2,3])
            # self.loss_b = tf.reduce_mean(
            #     intersection / (tf.reduce_sum(self.masks_predictions, axis=[1,2,3]) + tf.reduce_sum(self.masks, axis=[1,2,3]) - intersection)
            # )
            #loss_b = tf.losses.mean_squared_error(self.masks, self.masks_predictions_unrounded, scope="loss_b")

            loss = loss_a + self.loss_b
            global_step = tf.train.create_global_step()
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.labels_predictions` of shape [None] and type tf.int64
            # - mask predictions are stored in `self.masks_predictions` of shape [None, 28, 28, 1] and type tf.float32
            #   with values 0 or 1

            # Summaries
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.labels_predictions), tf.float32))
            only_correct_masks = tf.where(tf.equal(self.labels, self.labels_predictions),
                                          self.masks_predictions, tf.zeros_like(self.masks_predictions))
            # only_correct_masks = self.masks_predictions
            intersection = tf.reduce_sum(only_correct_masks * self.masks, axis=[1,2,3])
            self.iou = tf.reduce_mean(
                intersection / (tf.reduce_sum(only_correct_masks, axis=[1,2,3]) + tf.reduce_sum(self.masks, axis=[1,2,3]) - intersection)
            )

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/lossB", self.loss_b),
                                           tf.contrib.summary.scalar("train/accuracy", accuracy),
                                           tf.contrib.summary.scalar("train/iou", self.iou),
                                           tf.contrib.summary.image("train/images", self.images),
                                           tf.contrib.summary.image("train/masks", self.masks_predictions)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset+"/loss", loss),
                                               tf.contrib.summary.scalar(dataset+"/accuracy", accuracy),
                                               tf.contrib.summary.scalar(dataset+"/iou", self.iou),
                                               tf.contrib.summary.image(dataset+"/images", self.images),
                                               tf.contrib.summary.image(dataset+"/masks", self.masks_predictions)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels, masks):
        self.session.run([self.training, self.summaries["train"], self.loss_b],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})


    def evaluate(self, dataset, images, labels, masks):
        iou, _ = self.session.run([self.iou, self.summaries[dataset]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: False})
        return iou

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
                                {self.images: images, self.is_training: False, self.labels: np.zeros(images.shape[0], np.int64)})


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
    parser.add_argument("--encoder", default=None, type=str)
    parser.add_argument("--classification-decoder", default=None, type=str)
    parser.add_argument("--mask-decoder", default=None, type=str)
    args = parser.parse_args()

    # Create logdir name
    experiment_name = "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in sorted(vars(args).items()))).replace("/", "-")
    )
    args.logdir = "logs/" + experiment_name
    if not os.path.exists("logs"): os.mkdir("logs")  # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("fashion-masks-train.npz")
    dev = Dataset("fashion-masks-dev.npz")
    test = Dataset("fashion-masks-test.npz")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    max_iou = 0
    for i in range(args.epochs):
        while not train.epoch_finished():
            images, labels, masks = train.next_batch(args.batch_size)
            network.train(images, labels, masks)

        iou = network.evaluate("dev", dev.images, dev.labels, dev.masks)
        if iou > max_iou and iou > 0.912:
            labels, masks = network.predict(test.images)
            with open("results-{}.txt".format(experiment_name), "w") as test_file:
                for i in range(len(labels)):
                    print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)

    # labels, masks = network.predict(test.images)
    # with open("results-{}.txt".format(experiment_name), "w") as test_file:
    #     for i in range(len(labels)):
    #         print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)
