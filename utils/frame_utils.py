import numpy as np
from os.path import *
#from scipy.misc import imread
import cv2
from . import flow_utils 

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = cv2.imread(file_name)
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []
def fill_gen(img_shape):
    '''
    Generate a flow target containing zeros for testing. 
    Input:
        img: Input image shape
    Output:
        numpy array of all zeros
    '''
    return np.zeros(img_shape, dtype = np.float32)
