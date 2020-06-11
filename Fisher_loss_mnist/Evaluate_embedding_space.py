
import tensorflow as tf
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import dataset_characteristics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
import itertools


class Evaluate_embedding_space():

    def __init__(self, checkpoint_dir, model_dir_):
        self.checkpoint_dir = checkpoint_dir
        self.model_dir_ = model_dir_
        self.batch_size = 32
        self.n_samples = 100
        self.feature_space_dimension = 128

    def embed_the_data(self, X, labels, siamese, path_save_embeddings_of_test_data):
        print("Embedding the data....")

        saver_ = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            succesful_load, latest_epoch = self.load_network_model(saver_=saver_, session_=sess, checkpoint_dir=self.checkpoint_dir,
                                                                model_dir_=self.model_dir_)
            assert (succesful_load == True)

            X = self.normalize_images(X)

            test_feed_dict = {
                siamese.x1: X
            }
            embedding = sess.run(siamese.o1, feed_dict=test_feed_dict)
            if not os.path.exists(path_save_embeddings_of_test_data+"numpy\\"):
                os.makedirs(path_save_embeddings_of_test_data+"numpy\\")
            np.save(path_save_embeddings_of_test_data+"numpy\\embedding.npy", embedding)
            np.save(path_save_embeddings_of_test_data+"numpy\\labels.npy", labels)
            if not os.path.exists(path_save_embeddings_of_test_data+"plots\\"):
                os.makedirs(path_save_embeddings_of_test_data+"plots\\")
            # plt.figure(200)
            plt = self.plot_embedding_of_points(embedding=embedding, labels=labels, n_samples_plot=2000)
            plt.savefig(path_save_embeddings_of_test_data+"plots\\" + 'embedding.png')
            plt.clf()
            plt.close()
        return embedding, labels

    def normalize_images(self, X_batch):
        # also see normalize_images() method in Utils.py
        X_batch = X_batch * (1. / 255) - 0.5
        return X_batch

    def plot_embedding_of_points(self, embedding, labels, n_samples_plot=None):
        n_samples = embedding.shape[0]
        if n_samples_plot != None:
            indices_to_plot = np.random.choice(range(n_samples), min(n_samples_plot, n_samples), replace=False)
            embedding_sampled = embedding[indices_to_plot, :]
        else:
            indices_to_plot = [i for i in range(n_samples)]
            embedding_sampled = embedding
        if embedding.shape[1] == 2:
            pass
        else:
            embedding_sampled = umap.UMAP(n_neighbors=500).fit_transform(embedding_sampled)
        n_points = embedding.shape[0]
        # n_points_sampled = embedding_sampled.shape[0]
        labels_sampled = labels[indices_to_plot]
        _, ax = plt.subplots(1, figsize=(14, 10))
        classes = dataset_characteristics.get_class_names()
        n_classes = len(classes)
        plt.scatter(embedding_sampled[:, 0], embedding_sampled[:, 1], s=10, c=labels_sampled, cmap='Spectral',
                    alpha=1.0)
        # plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(n_classes + 1) - 0.5)
        cbar.set_ticks(np.arange(n_classes))
        cbar.set_ticklabels(classes)
        return plt

    def load_network_model(self, saver_, session_, checkpoint_dir, model_dir_):
        # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir_)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver_.restore(session_, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            latest_epoch = int(ckpt_name.split("-")[-1])
            return True, latest_epoch
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def classify_with_1NN(self, embedding, labels, path_to_save):
        print("KNN on embedding data....")
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        neigh = KNeighborsClassifier(n_neighbors=2)   #--> it includes itself too
        neigh.fit(embedding, labels)
        y_pred = neigh.predict(embedding)
        accuracy_test = accuracy_score(y_true=labels, y_pred=y_pred)
        conf_matrix_test = confusion_matrix(y_true=labels, y_pred=y_pred)
        self.save_np_array_to_txt(variable=np.asarray(accuracy_test), name_of_variable="accuracy_test", path_to_save=path_to_save)
        self.save_variable(variable=accuracy_test, name_of_variable="accuracy_test", path_to_save=path_to_save)
        # self.plot_confusion_matrix(confusion_matrix=conf_matrix_test, class_names=[str(class_index+1) for class_index in range(n_classes)],
        #                            normalize=True, cmap="gray_r", path_to_save=path_to_save, name="test")
        self.plot_confusion_matrix(confusion_matrix=conf_matrix_test, class_names=[str(class_index) for class_index in range(n_classes)],
                                   normalize=True, cmap="gray_r", path_to_save=path_to_save, name="test")

    def plot_confusion_matrix(self, confusion_matrix, class_names, normalize=False, cmap="gray", path_to_save="./", name="temp"):
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            pass
            # print('Confusion matrix, without normalization')
        # print(cm)
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        # plt.colorbar()
        tick_marks = np.arange(len(class_names))
        # plt.xticks(tick_marks, class_names, rotation=45)
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks, class_names)
        # tick_marks = np.arange(len(class_names) - 1)
        # plt.yticks(tick_marks, class_names[1:])
        fmt = '.2f' if normalize else 'd'
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        plt.ylabel('true distortion type')
        plt.xlabel('predicted distortion type')
        n_classes = len(class_names)
        plt.ylim([n_classes - 0.5, -0.5])
        plt.tight_layout()
        # plt.show()
        plt.savefig(path_to_save + name + ".png")
        plt.clf()
        plt.close()

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(
                path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))
