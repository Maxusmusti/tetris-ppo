import numpy as np
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    """
    Changes rgb image to grayscale
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def bw_pixel(gray):
    """
    Changes a grayscale pixel to only black or white
    """
    return (gray if gray == 0 else 1)

def crop_clean_state(state):
    """
    Crops only the tetris board from the state image and makes it grayscale [0-1]
    """
    cropped = state[46:209, 95:176]
    cropped = cropped / 255.0
    gray_cropped = rgb2gray(cropped)
    bw_vec = np.vectorize(bw_pixel)
    bw_cropped = np.array(list(map(bw_vec, gray_cropped)))
    plt.imshow(bw_cropped, interpolation='nearest', cmap="gray")
    plt.show()
    return bw_cropped

def extra_feats(info):
    """
    Extract the additional feature vector from the info dict
    """
    # TODO - Choose features
    return np.array([])
