import numpy as np
from numpy import matlib

def sceneflowconstruct(of, dt0, dt1):
    # Calculate depth using stereo baseline
    # Use camera matrix to get coordinates in the world frame
    # Use disparity change for sce flow
    # Plot three dimensional motion vectors
    focal_length = 1050.0
    baseline = 1.0

    # row = np.arange(540)
    # px = np.transpose(matlib.repmat(row, 960, 1))
    #
    # column = np.ax = range(960)
    # py = matlib.repmat(column, 540, 1)
    #
    # px_offset = 0
    # py_offset = 0
    # u = of[:, :, 0]  # Optical flow in horizontal direction
    # v = of[:, :, 1]  # optical flow in vertical direction
    #
    # z0 = (focal_length * baseline) / dt0
    # x0 = np.multiply((px - px_offset), z0[:, :, 0]) / focal_length
    # y0 = np.multiply((py - py_offset), z0[:, :, 0]) / focal_length
    #
    # # print(x0.shape, y0.shape, z0.shape)
    #
    # z1 = (focal_length * baseline) / dt1
    # x1 = np.multiply((px + u - px_offset), z1[:, :, 0]) / focal_length
    # y1 = np.multiply((py + v - py_offset), z1[:, :, 0]) / focal_length
    #
    # # print(x1.shape, y1.shape, z1.shape)
    #
    #
    # # Scene flow vectors
    #
    # dX = np.float32(x1 - x0)
    # dY = np.float32(y1 - y0)
    # dZ = np.float32(z1 - z0)
    #
    # scene_flow = np.dstack((dX, dY, dZ))

    # Scene flow as 4D vector
    sf = np.dstack((of[:, :, :2], dt0, dt1))

    return sf