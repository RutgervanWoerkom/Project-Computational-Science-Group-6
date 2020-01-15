import numpy as np
import random
import cv2

def generate_map_australia(out_x, out_y):
    """
    Create a vegetated map in the form of a numpy array of specified dimension (out_x, out_y).
    A recommended ratio is 11/9
    
    0: Ocean
    1: Forest
    2: Scrubland
    3: Woodland
    4: Grassland
    5: Desert
    """
    # import vegetation image
    veg = np.load('../datasets/processed/australia-vegetation.npy')

    # import mask image
    mask = np.load('../datasets/processed/australia-mask.npy')

    out_im = np.zeros((out_y, out_x))
    
    # Mask 
    veg = np.where(veg != 0, veg, 1)    
    mask = cv2.resize(mask, (out_x, out_y), interpolation=cv2.INTER_NEAREST)
    veg = cv2.resize(veg, (out_x, out_y), interpolation=cv2.INTER_NEAREST)
    veg = np.ma.masked_where(mask == 0, veg)
    
    assert len(np.unique(veg)) == 6, 'Provide valid vegetation image'
    assert len(np.unique(mask)) == 2, 'Provide valid mask image'

    for (i, p) in [(1, 0.1), (2, 0.3), (3, 0.5), (4, 0.7), (5, 0.9)]:
        for (x,y) in np.argwhere(veg == i):
            out_im[x, y] = (1 if random.random() < p else 2)
        
    return out_im