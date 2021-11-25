# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 20:09:09 2021

@author: kylei
"""
#%%
from google_images_search import GoogleImagesSearch
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

global CX_ID, API_KEY, LABEL

# Read more here on how to get your own keys:
# https://pypi.org/project/Google-Images-Search/

CX_ID = ""  # Custom search engine key
API_KEY = ""  # Google Cloud API key
LABEL = 0


def image_search(query, query_num):
    # Holds the image as a byte stream
    my_bytes_io = BytesIO()

    gis = GoogleImagesSearch(API_KEY, CX_ID)

    search_params = {
        "q": query,
        "num": query_num,
        "fileType": "jpg|png",
    }

    img_array = []
    gis.search(search_params)

    it = 0
    for image in gis.results():
        # Here we tell the BytesIO object to go back to address 0
        my_bytes_io.seek(0)

        # Take raw image data
        raw_image_data = image.get_raw_data()

        # This function writes the raw image data to the object
        image.copy_to(my_bytes_io, raw_image_data)

        # Or without the raw data which will be automatically taken
        # Inside the copy_to() method
        image.copy_to(my_bytes_io)

        # We go back to address 0 again so PIL can read it from start to finish
        my_bytes_io.seek(0)

        # Create a temporary image object
        try:
            temp_img = Image.open(my_bytes_io)

            # Downloads the original image for view later
            image.download(r"./images")

            temp_img = temp_img.resize(size=(250, 250))  # Change size to 250px by 250px

            # temp_img = temp_img.convert(mode="L") # Monochrome (1 channel)
            # temp_img = temp_img.convert(mode="CMYK") # Cyan, Magenta, Yellow, Black (4 channels)
            temp_img = temp_img.convert(mode="RGB")  # RGB (3 channels)

            # Convert the image into a matrix representation (features)
            numpydata = np.asarray(temp_img)

            # Plot the image
            plt.imshow(numpydata)

            # Append to data array
            img_array.append(numpydata)

            img_array_np = np.array(img_array)
            img_array_np = img_array_np / 255.0
            it += 1
            print(it)

        except Exception as e:
            print(e)
            print("Error on converting image")

    labels = np.full((query_num, 1), LABEL)

    return img_array_np, labels


#%%
if __name__ == "__main__":
    img_array_np0, labels0 = image_search("cat", 10)
    LABEL += 1
    img_array_np1, labels1 = image_search("stop sign", 10)
    LABEL += 1
    img_array_np2, labels2 = image_search("car", 10)
    LABEL += 1
    img_array_np3, labels3 = image_search("man", 10)

    # Stack labels and image matrices row-wise
    labels = np.vstack((labels0, labels1, labels2, labels3))
    data = np.vstack((img_array_np0, img_array_np1, img_array_np2, img_array_np3))

    # Save data as pickeled numpy arrays
    np.save(r"./images/data", data)
    np.save(r"./images/labels", labels)

# %%
