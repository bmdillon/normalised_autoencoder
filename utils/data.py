import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import gaussian_filter1d

class GaussianFilter:
    def __init__(self, sigma):
        self.sigma = sigma
        pass
    def __call__(self, img):
        img = img.reshape(-1,40,40)
        for i in (0,1):
            img = gaussian_filter1d(img, self.sigma, axis=i)
        img = img.reshape(-1,1600)
        return img

class NAEDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
            if isinstance( self.transform, GaussianFilter ):
                x = x[0,:]
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x

def load_data(path, batch_size, gfilter=False, gfsigma=2.0):
    data = torch.load(path)  # assumes a single tensor
    if gfilter:
        Gtransform = GaussianFilter( sigma=gfsigma )
        dataset = NAEDataset(data, transform=Gtransform)
    else:
        dataset = NAEDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_val(path, batch_size, gfilter=False, gfsigma=2.0):
    data, labels = torch.load(path)  # assumes tuple (data, labels)
    if gfilter:
        Gtransform = GaussianFilter( sigma=gfsigma )
        dataset = NAEDataset(data, labels=labels, transform=Gtransform)
    else:
        dataset = NAEDataset(data, labels=labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
