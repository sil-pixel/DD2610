# Transforms
import numpy as np
import PIL
import torch


class ArrayNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def __call__(self, arr: np.ndarray) -> np.ndarray:
        assert isinstance(arr, np.ndarray), "Input should be ndarray, got {}.".format(type(arr))
        assert arr.ndim >= 3, "Expected array to be image of size (*, H, W, C). Got {}.".format(arr.shape)
        
        dtype = arr.dtype
        mean = np.asarray(self.mean, dtype=dtype)
        std = np.asarray(self.std, dtype=dtype)
        if (std == 0).any():
            raise ValueError("std evaluated to zero after conversion to {}".format(dtype))
        if mean.ndim == 1:
            mean = mean.reshape(1, 1, -1)
        if std.ndim == 1:
            std = std.reshape(1, 1, -1)
        arr -= mean
        arr /= std
        return arr


class ToArray(torch.nn.Module):
    dtype = np.float32
    
    def __call__(self, x):
        assert isinstance(x, PIL.Image.Image)
        x = np.asarray(x, dtype=self.dtype)
        x /= 255.0
        return x