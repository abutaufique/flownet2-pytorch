from utils.flow_utils import readFlow
from os.path import join
import glob
import numpy as np
import cv2
def Visualize(folder):
    flows = sorted(glob.glob(join(folder, '*.flo')))
    for item in flows:
        flow = readFlow(item).astype(np.float32)
        mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        mag = (mag - mag.min()) / (mag.max() - mag.min())
        cv2.imshow('flow', mag)
        cv2.waitKey(1000)

if __name__ == '__main__':
    directory = './work/inference/run.epoch-0-flow-field/'
    Visualize(directory)
