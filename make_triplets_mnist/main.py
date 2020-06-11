
import numpy as np
# import pandas as pd
from random import shuffle
# from skimage.morphology import binary_erosion
import utils
import time
from PIL import Image
# import PIL
import glob
import matplotlib.pyplot as plt
import os, os.path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle


def main():
    dataset = "MNIST"  # MNIST, CIFAR10
    # n_triplets = 22528
    n_triplets = 500
    path_save_images = "./triplets/" + dataset + "/images/"
    path_save_labels = "./triplets/" + dataset + "/labels/"
    path_save_tfrecord = "./triplets/" + dataset + "/tfrecord/"

    if dataset == "MNIST":
        X_train, X_test, y_train, y_test = read_MNIST_dataset()
    elif dataset == "CIFAR10":
        X_train, X_test, y_train, y_test = read_CIFAR10_dataset()

    X_train_classes, X_test_classes, y_train_classes, y_test_classes = separate_classes(X_train, X_test, y_train, y_test)
    make_triplets(X_train_classes, y_train_classes, path_save_images, path_save_labels, path_save_tfrecord,
                  n_triplets, tfrecord_name='triplets.tfrecords')

def make_triplets(X_train_classes, y_train_classes, path_save_images, path_save_types, path_save_tfrecord, n_triplets, tfrecord_name='triplets.tfrecords'):
    if not os.path.exists(path_save_tfrecord):
        os.makedirs(path_save_tfrecord)
    with tf.io.TFRecordWriter(path_save_tfrecord + tfrecord_name) as writer:
        for triplet_index in range(n_triplets):
            if triplet_index % 20 == 0:
                print("processing triplet " + str(triplet_index) + "....")
            anchor, neighbor, distant, anchor_label, distant_label = extract_one_triplet(X_train_classes, y_train_classes)
            save_triplets_as_numpy(anchor, neighbor, distant, anchor_label, distant_label, path_save_images, path_save_types, triplet_index)
            save_triplets_as_tfrecord(anchor, neighbor, distant, anchor_label, distant_label, writer)

def extract_one_triplet(X_train_classes, y_train_classes):
    n_classes = len(X_train_classes)
    random_class_anchor, random_class_distant = 0, 0
    while random_class_anchor == random_class_distant:
        random_class_anchor = np.random.randint(n_classes, size=1)[0]
        random_class_distant = np.random.randint(n_classes, size=1)[0]
    X_train_anchor = X_train_classes[random_class_anchor]
    X_train_distant = X_train_classes[random_class_distant]
    ##### anchor and neighbor:
    n_samples_in_class_anchor = X_train_anchor.shape[0]
    random_anchor_index, random_neighbor_index = 0, 0
    while random_anchor_index == random_neighbor_index:
        random_anchor_index = np.random.randint(n_samples_in_class_anchor, size=1)[0]
        random_neighbor_index = np.random.randint(n_samples_in_class_anchor, size=1)[0]
    anchor = X_train_anchor[random_anchor_index, :, :].reshape((-1, 1))
    neighbor = X_train_anchor[random_neighbor_index, :, :].reshape((-1, 1))
    anchor_label = y_train_classes[random_class_anchor]
    ##### distant:
    n_samples_in_class_distant = X_train_distant.shape[0]
    random_distant_index = np.random.randint(n_samples_in_class_distant, size=1)[0]
    distant = X_train_distant[random_distant_index, :, :].reshape((-1, 1))
    distant_label = y_train_classes[random_class_distant]
    return anchor, neighbor, distant, anchor_label, distant_label

def save_triplets_as_numpy(anchor, neighbor, distant, anchor_label, distant_label, path_save_images, path_save_labels, triplet_index):
    # anchor:
    save_numpy(path=path_save_images, name_=str(triplet_index)+"anchor", array_=anchor)
    save_numpy(path=path_save_labels, name_=str(triplet_index)+"anchor_label", array_=np.array([anchor_label]))
    # neighbor:
    save_numpy(path=path_save_images, name_=str(triplet_index) + "neighbor", array_=neighbor)
    save_numpy(path=path_save_labels, name_=str(triplet_index) + "neighbor_label", array_=np.array([anchor_label]))
    # distant:
    save_numpy(path=path_save_images, name_=str(triplet_index) + "distant", array_=distant)
    save_numpy(path=path_save_labels, name_=str(triplet_index) + "distant_label", array_=np.array([distant_label]))

def save_triplets_as_tfrecord(anchor, neighbor, distant, anchor_label, distant_label, writer):
    # anchor:
    anchor_bytes = anchor.tostring()
    # neighbor:
    neighbor_bytes = neighbor.tostring()
    # distant:
    distant_bytes = distant.tostring()
    # wrapping:
    data = \
        {
            'image_anchor': wrap_bytes(anchor_bytes),
            'image_neighbor': wrap_bytes(neighbor_bytes),
            'image_distant': wrap_bytes(distant_bytes),
            'label_anchor': wrap_int64(anchor_label),
            'label_neighbor': wrap_int64(anchor_label),
            'label_distant': wrap_int64(distant_label)
        }
    feature = tf.train.Features(feature=data)
    example = tf.train.Example(features=feature)
    serialized = example.SerializeToString()
    writer.write(serialized)

def count_files_in_folder(path_):
    # simple version for working with CWD
    return len([name for name in os.listdir(path_) if os.path.isfile(os.path.join(path_, name))])

def save_numpy(path, name_, array_):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+name_+".npy", array_)

def read_images(path, image_format="tif"):
    image_list = []
    for filename in glob.glob(path+"*."+image_format):
        # print("processing file: " + str(filename))
        im = Image.open(filename)
        image_list.append(im)
    return image_list

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save_numpy_array(path_to_save, arr_name, arr):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    np.save(path_to_save+arr_name+".npy", arr)

def read_MNIST_dataset():
    # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data?version=stable
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    #######################
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # plt.imshow(X_train[0, :, :])
    # plt.show()
    # input("hi")
    #######################
    return X_train, X_test, y_train, y_test

def read_CIFAR10_dataset():
    # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data?version=stable
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #######################
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    #######################
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # plt.imshow(X_train[0, :, :])
    # plt.show()
    # input("hi")
    #######################
    return X_train, X_test, y_train, y_test

def separate_classes(X_train, X_test, y_train, y_test):
    unique_labels = np.unique(y_train)
    unique_labels = np.sort(unique_labels)
    n_classes = len(unique_labels)
    X_train_classes = [None] * n_classes
    y_train_classes = [None] * n_classes
    X_test_classes = [None] * n_classes
    y_test_classes = [None] * n_classes
    for class_index in range(n_classes):
        class_label = unique_labels[class_index]
        X_train_classes[class_index] = X_train[y_train == class_label, :, :]
        y_train_classes[class_index] = y_train[y_train == class_label][0]
        X_test_classes[class_index] = X_test[y_test == class_label, :, :]
        y_test_classes[class_index] = y_test[y_test == class_label][0]
    return X_train_classes, X_test_classes, y_train_classes, y_test_classes


if __name__ == "__main__":
    main()
