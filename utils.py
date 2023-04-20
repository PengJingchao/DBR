import logging
import numpy as np
import torch



def calculatePreRecF1Measure(output_image, gt_image, thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image > thre
    gt_bin = gt_image
    recall = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(gt_bin))
    precision = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(out_bin))
    F1 = 2 * recall * precision / np.maximum(0.001, recall + precision)
    IOU = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(out_bin) + np.sum(gt_bin) - np.sum(gt_bin * out_bin))
    return precision, recall, F1, IOU
