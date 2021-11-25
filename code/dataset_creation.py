# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:37:04 2021

@author: kylei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

if __name__ == "__main__":
    n_classes = 2
    data, labels = make_blobs(
        n_samples=1000,
        centers=n_classes,
        random_state=10,
        center_box=(-10.0, 10.0), # Changes center of each sample cluster
        cluster_std=2.5, # Changes sample spread
    )

    toy_dataset = pd.DataFrame(data=data)
    toy_dataset[2] = labels

    # Plotting
    fig, ax = plt.subplots()

    np.random.seed(192)  # Set seed for colours

    # Randomly generate a colour for each class
    colours = []
    for i in range(n_classes):
        colours.append(
            [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
        )

    for label in range(n_classes):
        ax.scatter(
            x=data[labels == label, 0],
            y=data[labels == label, 1],
            color=colours[label],
            s=40,
            label="Class {c}".format(c=label),
        )

    ax.set(xlabel="X", ylabel="Y", title="Toy Example")

    ax.legend(loc="upper right")
    plt.show()