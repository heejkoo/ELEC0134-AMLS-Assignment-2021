import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(os.path.normpath(dir))
sys.path.insert(0, main_dir)

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf

from models import build_convnet, build_pretrained_vgg
from utils import data_preprocessing, set_global_determinism

# For Reproducibility
seed=42
set_global_determinism(seed)


class B1(object):
    """Task B1"""

    def __init__(self, train_set, val_set, test_set, model, gap, init, num_epochs, freeze=False):

        # Task
        self.sort = 'B1'

        # Data Loader
        self.train_set = train_set
        self.train_images, self.train_labels = train_set
        self.val_set = val_set
        self.val_images, self.val_labels = val_set
        self.test_set = test_set
        self.test_images, self.test_labels = test_set

        # Training Configurations
        self.in_height = 72
        self.in_width = 72
        self.num_classes = 5
        self.optimizer = "adam"
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = "accuracy"
        self.freeze = freeze
        self.init = init
        self.num_epochs = num_epochs
        self.gap = gap

        # Path
        self.figure_path = "../results/fig/"
        self.checkpoint_path = "../results/checkpoint/"
        self.make_directory(self.figure_path)
        self.make_directory(self.checkpoint_path)

        # Model
        self.model = model
        if self.model == 'custom':
            self.model = self.build_custom_cnn()
        elif self.model == 'vgg':
            self.model = self.build_pretrained_vgg()

    def make_directory(self, path):
        """Make Directory"""
        if not os.path.exists(path):
            os.makedirs(path)

    def build_custom_cnn(self):
        """Build Custom Models"""
        self.custom_cnn = build_convnet(self.in_height, self.in_width, self.num_classes, self.gap, self.init)
        self.custom_cnn.summary()
        return self.custom_cnn

    def build_pretrained_vgg(self):
        """Build Pretrained VGGNet"""
        self.vgg = build_pretrained_vgg(self.in_height, self.in_width, self.num_classes, self.gap, self.freeze)
        self.vgg.summary()
        return self.vgg

    def plot(self):
        """Plot after training"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("{} {} Training and Val Accuracy and Loss".format(self.sort, str(self.model.name)))

        ax1.plot(self.history.history["accuracy"], label="accuracy")
        ax1.plot(self.history.history["val_accuracy"], label="val_accuracy")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")
        ax1.grid()
        ax1.legend(loc="best")
        ax1.title.set_text("Accuracy")

        ax2.plot(self.history.history["loss"], label="loss")
        ax2.plot(self.history.history["val_loss"], label="val_loss")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.grid()
        ax2.legend(loc="best")
        ax2.title.set_text("Loss")

        plt.savefig(os.path.join(self.figure_path, "{} {} Training and Val Accuracy and Loss.png".format(self.sort, str(self.model.name))))

    def confusion_matrix(self):
        """Confusion Matrix"""
        conf_matrix = confusion_matrix(self.test_labels, self.pred)
        conf_matrix = pd.DataFrame(conf_matrix, index=[i for i in "01234"], columns=[i for i in "01234"])

        plt.figure(num='Confusion Matrix')

        if str(self.model.name).__contains__('Custom'):
            cnn = 'CustomCNN'
        elif str(self.model.name).__contains__('VGG'):
            cnn = 'VGG'

        plt.title("{} {} Confusion Matrix.png".format(self.sort, cnn), fontsize=14)

        ax = sns.heatmap(conf_matrix, fmt='d', cmap='YlGnBu', annot=True)
        ax.set_xlabel('True Class', fontsize=14)
        ax.set_ylabel('Predicted Class', fontsize=14)
        plt.savefig(os.path.join(self.figure_path, "{} {} Confusion Matrix.png".format(self.sort, str(self.model.name))))

    def train(self):
        """Train"""

        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics]
        )

        self.save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_path, '{}_BEST_{}.ckpt'.format(self.sort, str(self.model.name))),
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            mode='auto',
            save_freq='epoch',
            verbose=2
        )

        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True)

        self.history = self.model.fit(
            self.train_images,
            self.train_labels,
            epochs=self.num_epochs,
            validation_data=(self.val_images, self.val_labels),
            callbacks=[self.save_best, self.early_stopping]
        )

        self.plot()

    def test(self):
        """Test"""

        self.model.load_weights(
            filepath=os.path.join(self.checkpoint_path, "{}_BEST_{}.ckpt".format(self.sort, str(self.model.name))))

        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics]
        )

        test_accuracy = self.model.evaluate(self.test_images, self.test_labels, verbose=2)[1]
        print("Accuracy  : {:.4f}".format(test_accuracy))

        self.pred = self.model.predict(self.test_images)
        self.pred = np.argmax(self.pred, axis=1)
        self.confusion_matrix()

        print("Precision : {:.4f}".format(precision_score(self.test_labels, self.pred, average="weighted")))
        print("Recall    : {:.4f}".format(recall_score(self.test_labels, self.pred, average="weighted")))
        print("F1 Score  : {:.4f}".format(f1_score(self.test_labels, self.pred, average="weighted")))

        return test_accuracy


if __name__ == "__main__":
    from config import *
    data_train, data_val, data_test = data_preprocessing('cartoon_set', 'face_shape')
    model_B1 = B1(data_train, data_val, data_test, config.model, config.gap, config.init, config.num_epochs, config.freeze)
    acc_B1_train = model_B1.train()
    acc_B1_test = model_B1.test()