import numpy as np
import random
from random import choices
import time

MAX_TRIAL = 500

def positive_or_negative():
    return 1 if random.random() < 0.5 else -1

def sample_anchor(msk_eroded,tile_size):
    try:
        counter = 0
        width, height = msk_eroded.shape
        x_anchor = -1
        y_anchor = -1
        while x_anchor<tile_size or x_anchor>width-tile_size or y_anchor<tile_size or y_anchor>=height-tile_size:
            x_anchor, y_anchor = choices(np.asarray(np.where(msk_eroded==1)).T)[0]
            counter += 1
            if counter>MAX_TRIAL:
                return None, None
        return y_anchor, x_anchor
    except:
        return None, None

def sample_neighbor(msk_shape, x_anchor, y_anchor, tile_size):
    try:
        width, height = msk_shape
        x_neighbor, y_neighbor = -1, -1
        if x_anchor == None or y_anchor == None:
            return None, None

        counter = 0
        while x_neighbor<tile_size or x_neighbor>=width-tile_size:
            counter =+ 1
            x_neighbor = np.random.randint(tile_size//2, 2*tile_size)*positive_or_negative() + x_anchor
            if counter>MAX_TRIAL:
                return None, None
        counter =+ 0        
        while y_neighbor<tile_size or y_neighbor>=height-tile_size:
            counter =+ 1
            y_neighbor = np.random.randint(tile_size//2, 2*tile_size)*positive_or_negative() + y_anchor
            if counter>MAX_TRIAL:
                return None, None
        return y_neighbor, x_neighbor
    except:
        return None, None

def sample_distant_same(msk_eroded, x_anchor, y_anchor, neighborhood=1024):
    try:
        x = np.arange(0, msk_eroded.shape[1])
        y = np.arange(0, msk_eroded.shape[0])
        proximity = (x[np.newaxis,:]-x_anchor)**2 + (y[:,np.newaxis]-y_anchor)**2 < neighborhood**2
        msk_eroded[proximity] = 0
        return choices(np.asarray(np.where(msk_eroded==1)).T)[0]
    except:
        return None, None

def sample_distant_diff(msk_eroded,tile_size):
    return sample_anchor(msk_eroded,tile_size)

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))