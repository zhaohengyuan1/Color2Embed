import numpy as np
import data.thinplate as tps
import cv2
import random
import math

# Reference : https://github.com/cheind/py-thin-plate-spline

def tps_transform(img, dshape=None):

    while True:
        point1 = round(random.uniform(0.3, 0.7), 2)
        point2 = round(random.uniform(0.3, 0.7), 2)
        range_1 = round(random.uniform(-0.25, 0.25), 2)
        range_2 = round(random.uniform(-0.25, 0.25), 2)
        if math.isclose(point1 + range_1, point2 + range_2):
            continue
        else:
            break

    c_src = np.array([
        [0.0, 0.0],
        [1., 0],
        [1, 1],
        [0, 1],
        [point1, point1],
        [point2, point2],
    ])

    c_dst = np.array([
        [0., 0],
        [1., 0],
        [1, 1],
        [0, 1],
        [point1 + range_1, point1 + range_1],
        [point2 + range_2, point2 + range_2],
    ])

    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

