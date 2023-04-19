import os.path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import random as rng

rng.seed(12345)

'''
centers:中心坐标(|x|,|y|)
patch_centers：每个patch内的中心坐标(x,y)(<16)
multiply_four_edge_distances:x*y*(16-x)*(16-y)
patch_*:考虑了batch维度
'''


def find_centers(binary_map):
    canny_output = cv2.Canny(binary_map, 100, 200)

    contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    centers_vectors = np.zeros((len(contours), 2, 16))
    # print(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

        # # show
        # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        # for i in range(len(contours)):
        #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        #     cv2.drawContours(drawing, contours_poly, i, color)
        #     cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        #
        # cv2.imshow('Contours', drawing)
        # cv2.waitKey(0)
        centers_int = np.divmod(np.round(centers[i]), (16, 16))[1].astype('int')
        a = np.array([[1 / 1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4,
                       1 / 3, 1 / 2]])
        # centers_vector = a / a.sum()
        centers_vector = (a - a.mean()) / a.std()
        centers_vector_0 = np.roll(centers_vector, centers_int[0])
        centers_vector_1 = np.roll(centers_vector, centers_int[1])
        centers_vectors[i, :, :] = np.append(centers_vector_0, centers_vector_1, axis=0)
        # centers_vectors = F.one_hot(torch.tensor(centers_int), 16)
    value = radius / np.sum(radius)
    centers_vector = np.einsum('i,ijk', value, centers_vectors)

    return centers_vector


def find_batch_centers(batch_binary_map):
    assert len(batch_binary_map.shape) != 2
    batch_centers_vectors = np.zeros((batch_binary_map.shape[0], 2, 16))
    for batch_index in range(batch_binary_map.shape[0]):
        binary_map = batch_binary_map[batch_index, :, :]
        canny_output = cv2.Canny(binary_map, 100, 200)

        contours, radius = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None] * len(contours)
        centers = [None] * len(contours)
        radius = [None] * len(contours)
        centers_vectors = np.zeros((len(contours), 2, 16))

        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

            centers_int = np.divmod(np.round(centers[i]), (16, 16))[1].astype('int')
            a = np.array([[1 / 1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5,
                           1 / 4, 1 / 3, 1 / 2]])
            # centers_vector = a / a.sum()
            '''np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))'''
            centers_vector = (a - a.mean()) / a.std()
            centers_vector_0 = np.roll(centers_vector, centers_int[0])
            centers_vector_1 = np.roll(centers_vector, centers_int[1])
            centers_vectors[i, :, :] = np.append(centers_vector_0, centers_vector_1, axis=0)
            # centers_vectors = F.one_hot(torch.tensor(centers_int), 16)
        value = radius / np.sum(radius)
        centers_vector = np.einsum('i,ijk', value, centers_vectors)
        batch_centers_vectors[batch_index, :, :] = centers_vector

    return batch_centers_vectors

