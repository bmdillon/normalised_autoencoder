import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

class GaussianFilter:
    def __init__(self, sigma):
        self.sigma = sigma
        pass

    def __call__(self, img):
        for i in (1,2):
            img = gaussian_filter1d(img, self.sigma, axis=i)
        return img
