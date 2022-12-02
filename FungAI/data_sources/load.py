import numpy as np
import os
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True # important to avoid an error (the truncated picture error)
from google.cloud import storage

from FungAI.ml.params import LOCAL_DATA_PATH

def load_local() :
    '''
    Function that take the local raw data, compress and return them as a tensor X and y.
    '''
    _X = []
    _y = []

    idx = 0
    for genus in os.listdir(LOCAL_DATA_PATH):
        for image in os.listdir(f"{LOCAL_DATA_PATH}/{genus}"):
            temp_img = Image.open(os.path.join(LOCAL_DATA_PATH, genus, image))
            if len(temp_img.shape) < 3 : # necessary because there is an image that has no RGB dimension (wtf ??)
                    continue
            trans_img = np.ndarray(temp_img.resize((100, 100))) # normlizing the pixels of images
            _X.append(trans_img)
            _y.append(genus)
            idx += 1

    X = np.concatenate(_X , axis = 0).reshape((idx, 100, 100, 3)) # putting an image as one item in the tensor
    y = np.array(_y)

    return X, y



def load_cloud() :
    '''❗️NOT WORKING❗️'''

    BUCKET_NAME = "zipped_mushrooms"

    storage_filename = "data/raw/train_1k.csv"
    local_filename = "train_1k.csv"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(storage_filename)
    blob.download_to_filename(local_filename)