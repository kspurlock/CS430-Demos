# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 23:54:51 2021

@author: kylei
"""
#%%
import tensorflow as tf
from tensorflow.keras import layers, models
import keras
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import KFold
from lime import lime_image
from skimage.segmentation import mark_boundaries

if __name__ == "__main__":
    data = np.load(r"./images/data.npy")
    data = data.reshape(-1, 250, 250, 3)
    labels = np.load(r"./images/labels.npy")

    # Provides plotting for the image classes
    class_names = ["cat", "stop sign", "car", "man"]

    seen_classes = []
    samples_idx = []

    for i in range(data.shape[0]):
        if len(samples_idx) == 4:
            break

        else:
            label = labels[i]

            if label not in seen_classes:
                samples_idx.append(i)
                seen_classes.append(label)

    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data[samples_idx[i]])
        plt.xlabel(class_names[seen_classes[i][0]])
    plt.show()

#%%
    cv = KFold(n_splits=5, random_state=None)  # Equivalent to 20% test size split
    split_dict = {}
    it = 0
    for train_ind, test_ind in cv.split(data):
        split_dict[it] = (train_ind, test_ind)
        it += 1

    choose_split = 2  # Select a split from KFolds, basically treating this as a train_test_split right now

    x_train, y_train = (
        data[split_dict[choose_split][0]],
        labels[split_dict[choose_split][0]],
    )
    x_test, y_test = (
        data[split_dict[choose_split][1]],
        labels[split_dict[choose_split][1]],
    )

#%% CNN training

    # Creating a convolutional model
    model = models.Sequential()
    model.add(layers.Conv2D(10, (3, 3), activation="relu", use_bias=True))
    model.add(layers.MaxPooling2D((4, 4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="relu", use_bias=True))
    model.add(layers.Dense(4, activation="softmax", use_bias=True))

    start = time.process_time()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train, y_train, epochs=20, batch_size=1, validation_data=(x_test, y_test)
    )

    stop = time.process_time() - start
    print(stop)

#%% Visualizing performance

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

#%% LIME explanations

    for i in range(len(class_names)):  # Want to find a
        explainer = lime_image.LimeImageExplainer()
        class_i_want = x_train[
            y_train.reshape(-1,)== i
        ]
        sample = np.random.randint(0, class_i_want.shape[0]) # Pick a random sample index
        explanation = explainer.explain_instance(
            x_train[i].astype("double"),
            model.predict,
            hide_color=0,
            num_samples=1000,
            top_labels=4,
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False,
        )  # Num features here are the super pixels
        plt.imshow(mark_boundaries(temp, mask))
        plt.title(class_names[explanation.top_labels[0]])

#%%
