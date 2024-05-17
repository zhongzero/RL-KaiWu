#!/usr/bin/env python
# -*- coding: utf-8 -*-


# 需要numpy 1.20.0或者以下版本
from numba import jit
from numba import cuda
import numpy as np


@jit
def neighbor_list(crd, neighbors, data_length, cutoff):
    """CPU based neighbor list calculation.
    """
    for i in range(data_length):
        for j in range(i+1, data_length):
            if np.linalg.norm(crd[i]-crd[j]) <= cutoff:
                neighbors[i][j] = 1
                neighbors[j][i] = 1
    return neighbors

@cuda.jit
def cuda_neighbor_list(crd, neighbors, cutoff):
    """GPU based neighbor list calculation.
    """
    i, j = cuda.grid(2)
    dis = ((crd[i][0]-crd[j][0])**2+\
           (crd[i][1]-crd[j][1])**2+\
           (crd[i][2]-crd[j][2])**2)**0.5
    neighbors[i][j] = dis <= cutoff[0] and dis > 0

'''
本代码用于验证机器的GPU是否可用
'''
if __name__ == '__main__':
    import time
    np.random.seed(1)

    atoms = 2**10
    cutoff = 0.5
    cutoff_cuda = cuda.to_device(np.array([cutoff]).astype(np.float32))
    crd = np.random.random((atoms,3)).astype(np.float32)
    crd_cuda = cuda.to_device(crd)
    adjacent = np.zeros((atoms, atoms)).astype(np.float32)
    adjacent_cuda = cuda.to_device(adjacent)
    time_c = 0.0
    time_g = 0.0

    for _ in range(100):
        time0 = time.time()
        adjacent_c = neighbor_list(crd, adjacent, atoms, cutoff)
        time1 = time.time()
        cuda_neighbor_list[(atoms, atoms), (1, 1)](crd_cuda,
                                                adjacent_cuda,
                                                cutoff_cuda)
        time2 = time.time()
        if _ != 0:
            time_c += time1 - time0
            time_g += time2 - time1
    
    print ('The total time cost of CPU with numba.jit is: {}s'.format(\
                                            time_c))
    print ('The total time cost of GPU with cuda.jit is: {}s'.format(\
                                            time_g))