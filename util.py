from time import perf_counter_ns as pcn
from scipy.signal import convolve2d
from torch.nn import Conv2d as conv2
import torch
from .. import juliaTestRobot

def nanosecond_to_milisecond(t):
    return t*1e-6


def convolve_data():
    matrix2 = torch.zeros(100,100)
    kernel = torch.ones([10,10])
    for nr, array in enumerate(matrix, start=0):
        convolved = convolve2d(array, kernel, mode="valid")
        matrix2[nr] = torch.from_numpy(convolved[::10, ::10]).flatten()