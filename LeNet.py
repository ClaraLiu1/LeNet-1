import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from collections import namedtuple

CHECKPOINT_DIR = os.path.join(os.getcwd(), 'pretrained_model')

def load_data(dataset_name):
    mnist = input_data.read_data_sets(dataset_name, reshape=False)

    dataset = {"train_images": mnist.train.images, "train_labels": mnist.train.labels,
               "validation_images": mnist.validation.images, "validation_labels": mnist.validation.labels,
               "test_images": mnist.test.images, "test_labels": mnist.test.labels}

    assert(len(dataset["train_images"]) == len(dataset["train_labels"]))
    assert(len(dataset["validation_images"]) == len(dataset["validation_labels"]))
    assert(len(dataset["test_images"]) == len(dataset["test_labels"]))

    print()
    print("DATASET INFO:")
    print("Image Shape: {}".format(dataset["train_images"][0].shape))
    print()
    print("Training Set:   {} samples".format(len(dataset["train_images"])))
    print("Validation Set: {} samples".format(len(dataset["validation_images"])))
    print("Test Set:       {} samples".format(len(dataset["test_images"])))
    return dataset


# we pad the data with two rows of zeros on the top and bottom,
# and two columns of zeros on the left and right (28+2+2 = 32).
def pad_dataset(dataset):
    x_train = dataset["train_images"]
    x_validation = dataset["validation_images"]
    x_test = dataset["test_images"]

    # Pad images with 0s
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_validation = np.pad(x_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    dataset["train_images"] = x_train
    dataset["validation_images"] = x_validation
    dataset["test_images"] = x_test

    print("Updated Image Shape: {}".format(dataset["train_images"][0].shape))
    return dataset


# Setup tensorflow
Parameters = namedtuple("Parameters", [
    # Data parameters
    "image_shape", "num_classes",
    # Training parameters
    "is_trained", "learning_rate",
    "epochs", "batch_size",
    "mu", "sigma",
    # Layers architecture
    "conv1_k", "conv1_d",
    "conv2_k", "conv2_d",
    "fc3_d",
    "fc4_d",
    "output_d"])


class LeNet(object):
    def __init__(self, params):
        # Feature and labels placeholder
        self.x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        self.y = tf.placeholder(tf.int32, (None, ))
        self.params = params
        self.one_hot_y = self.one_hot()
        self.logits = self.classifier()

    def one_hot(self):
        return tf.one_hot(self.y, self.params.num_classes)

    # Build the architecture
    def classifier(self):
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=self.params.mu, stddev=self.params.sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(self.x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        # Activation.
        conv1 = tf.nn.relu(conv1)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Layer 2: Convolutional. Output = 10x10x16.
        conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=self.params.mu, stddev=self.params.sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        # Activation.
        conv2 = tf.nn.relu(conv2)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten. Input = 5x5x16. Output = 400.
        fc1 = flatten(pool_2)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.params.mu, stddev=self.params.sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc1, fc1_w) + fc1_b

        # Activation.
        fc1 = tf.nn.relu(fc1)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self.params.mu, stddev=self.params.sigma))
        fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1, fc2_w) + fc2_b
        # Activation.
        fc2 = tf.nn.relu(fc2)

        # Layer 5: Fully Connected. Input = 84. Output = 10.
        fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=self.params.mu, stddev=self.params.sigma))
        fc3_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(fc2, fc3_w) + fc3_b
        return logits

    def cross_entropy_and_loss(self, x, y, logits):
        # Model Evaluatation
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        num_examples = len(x)
        total_accuracy = 0
        sess = tf.get_default_session()

        # Restoring the model if exists
        if 'checkpoint' in os.listdir(CHECKPOINT_DIR):
            tf.train.Saver().restore(sess=sess, save_path='./pretrained_model/lenet')
            print('Restoring the model was successful!!!\n')
        else:
            print("Oops! No model had been saved in %s." % str(CHECKPOINT_DIR))
            print("Restoring model failed!!!")

        for offset in range(0, num_examples, self.params.batch_size):
            batch_x, batch_y = x[offset:offset + self.params.batch_size], y[offset:offset + self.params.batch_size]
            accuracy = sess.run(accuracy_operation, feed_dict={self.x: batch_x, self.y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def train(self, x_train, y_train, x_valid, y_valid):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
        training_operation = optimizer.minimize(loss_operation)

        saver = tf.train.Saver()

        # Training
        if self.params.is_trained:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                num_examples = len(x_train)

                print("Training...")
                print()
                for i in range(self.params.epochs):
                    x_train, y_train = shuffle(x_train, y_train)
                    for offset in range(0, num_examples, self.params.batch_size):
                        end = offset + self.params.batch_size
                        batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                        sess.run(training_operation, feed_dict={self.x: batch_x, self.y: batch_y})

                    validation_accuracy = self.cross_entropy_and_loss(x_valid, y_valid, self.logits)
                    print("EPOCH {} ...".format(i + 1))
                    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                    print()

                # save_path = saver.save(sess, "/tmp/model.ckpt")
                # print("Model saved in path: %s" % save_path)
                saver.save(sess, './pretrained_model/lenet')
                print("Model saved")

    def evaluate(self, x_test, y_test):

        # Evaluate the Model
        with tf.Session() as sess:
            # tf.train.Saver().restore(sess, tf.train.latest_checkpoint('.'))
            # tf.train.Saver().restore(sess, './lenet')

            test_accuracy = self.cross_entropy_and_loss(x_test, y_test, self.logits)
            print("Test Accuracy = {:.3f}".format(test_accuracy))


if __name__ == "__main__":
    # Load Dataset
    mnist_dataset = pad_dataset(load_data("MNIST_data/"))
    X_train = mnist_dataset["train_images"]
    y_train = mnist_dataset["train_labels"]
    X_valid = mnist_dataset["validation_images"]
    y_valid = mnist_dataset["validation_labels"]
    X_test = mnist_dataset["test_images"]
    y_test = mnist_dataset["test_labels"]

    # Pre-process Data
    # Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train)

    # Initialize needed parameters
    parameters = Parameters(image_shape=(32, 32, 1),
                            num_classes=10,
                            is_trained=True,
                            learning_rate=0.001,
                            epochs=10,
                            batch_size=128,
                            # Hyperparameters
                            mu=0.0,
                            sigma=0.1,
                            conv1_d=6, conv1_k=5,
                            conv2_d=16, conv2_k=5,
                            fc3_d=120,
                            fc4_d=84,
                            output_d=10
                            )

    # Train LeNet model
    lenet = LeNet(parameters)
    lenet.train(x_train=X_train, y_train=y_train, x_valid=X_valid, y_valid=y_valid)

    # Evaluate LeNet model
    lenet.evaluate(x_test=X_test, y_test=y_test)
