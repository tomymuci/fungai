import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # important to avoid an error (the truncated picture error)

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
            temp_img = plt.imread( os.path.join(LOCAL_DATA_PATH, genus, image))
            if len(temp_img.shape) < 3 : # necessary because there is an image that has no RGB dimension (wtf ??)
                    continue
            trans_img = cv2.resize(temp_img, (100, 100), interpolation = cv2.INTER_AREA) # normlizing the pixels of images
            _X.append(trans_img)
            _y.append(genus)
            idx += 1

    X = np.concatenate(_X , axis = 0).reshape((idx, 100,100,3)) # putting an image as one item in the tensor
    y = np.array(_y)

    return X, y



def load_cloud() :
    pass
