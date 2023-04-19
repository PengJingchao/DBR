import cv2
import random
import numpy as np


def random_rotate_img_mask(img, mask):
    mask_rotated = np.zeros(1)
    while mask_rotated.sum() == 0:
        img_h, img_w = img.shape[0], img.shape[1]
        # 生成旋转参数
        h, w = random.randint(0, 255), random.randint(0, 255)  # 旋转中心
        d = random.uniform(-90, 90)  # 旋转度数
        n = random.uniform(0.8, 10)  # 放缩倍数
        M = cv2.getRotationMatrix2D((h, w), d, n)  # 旋转矩阵
        # 进行旋转
        img_rotated = cv2.warpAffine(img, M, (img_w, img_h))
        mask_rotated = cv2.warpAffine(mask, M, (img_w, img_h))

    # print(img_rotated.shape)
    # print(mask_rotated.shape)
    # print(mask_rotated.sum())
    # cv2.imshow('r1', img_rotated)
    # cv2.imshow('r2', mask_rotated)
    # cv2.waitKey(0)
    return img_rotated, mask_rotated


def random_crop_img_mask(img, mask):
    img_h, img_w = img.shape[0], img.shape[1]

    scale = random.uniform(0.4, 0.9)  # 裁剪比例
    height, width = int(img_h * scale), int(img_w * scale)  # 裁减区域尺寸
    x, y = random.randint(0, img_w - width), random.randint(0, img_h - height)  # 裁剪区域的起点

    img_croped = cv2.resize(img[y:y + height, x:x + width], (img_w, img_h))  # 裁减+调整回原图大小
    mask_croped = cv2.resize(mask[y:y + height, x:x + width], (img_w, img_h))  # 裁减+调整回原图大小
    # print(img_croped.shape)
    # print(mask_croped.shape)
    # print(mask_croped.sum())
    # cv2.imshow('r2', mask_croped)
    # cv2.imshow('r1', img_croped)
    # cv2.waitKey(0)
    return img_croped, mask_croped


# 镜像变换
def random_mirror_img_mask(img, mask):
    mode = random.randint(0, 1)  # mode = 1 水平翻转 mode = 0 垂直翻
    img_mirror = cv2.flip(img, mode)
    mask_mirror = cv2.flip(mask, mode)

    # print(img_mirror.shape)
    # print(mask_mirror.shape)
    # print(mask_mirror.sum())
    # cv2.imshow('r2', mask_mirror)
    # cv2.imshow('r1', img_mirror)
    # cv2.waitKey(0)
    return img_mirror, mask_mirror


# 仿射
def random_affine_img_mask(img, mask):
    img_h, img_w = img.shape[0], img.shape[1]
    mask_affined = np.zeros(1)
    point1 = np.float32([[2, 2], [3, 3.732], [4, 2]])
    while mask_affined.sum() == 0:
        x1 = random.uniform(0, 3)
        y1 = random.uniform(0, 3 - x1)
        x2 = random.uniform(0, 4)
        y2 = random.uniform(3.732, 4)
        x3 = random.uniform(3, 6)
        y3 = random.uniform(0, x3 - 3)
        point2 = np.float32([[x1, y1], [x2, y2], [x3, y3]])
        M = cv2.getAffineTransform(point1, point2)

        img_affined = cv2.warpAffine(img, M, (img_w, img_h))
        mask_affined = cv2.warpAffine(mask, M, (img_w, img_h))

        # print(img_affined.shape)
        # print(mask_affined.shape)
        # print(mask_affined.sum())
        # cv2.imshow('r2', mask_affined)
        # cv2.imshow('r1', img_affined)
        # cv2.waitKey(0)
    return img_affined, mask_affined


def random_rotate_img_mask_notargets(img, mask, notargets):
    mask_rotated = np.zeros(1)
    while mask_rotated.sum() == 0:
        img_h, img_w = img.shape[0], img.shape[1]
        # 生成旋转参数
        h, w = random.randint(0, 255), random.randint(0, 255)  # 旋转中心
        d = random.uniform(-90, 90)  # 旋转度数
        n = random.uniform(0.8, 10)  # 放缩倍数
        M = cv2.getRotationMatrix2D((h, w), d, n)  # 旋转矩阵
        # 进行旋转
        img_rotated = cv2.warpAffine(img, M, (img_w, img_h))
        mask_rotated = cv2.warpAffine(mask, M, (img_w, img_h))
        notargets_rotated = cv2.warpAffine(notargets, M, (img_w, img_h))

    # print(img_rotated.shape)
    # print(mask_rotated.shape)
    # print(mask_rotated.sum())
    # cv2.imshow('r1', img_rotated)
    # cv2.imshow('r2', mask_rotated)
    # cv2.waitKey(0)
    return img_rotated, mask_rotated, notargets_rotated


def random_crop_img_mask_notargets(img, mask, notargets):
    img_h, img_w = img.shape[0], img.shape[1]

    scale = random.uniform(0.4, 0.9)  # 裁剪比例
    height, width = int(img_h * scale), int(img_w * scale)  # 裁减区域尺寸
    x, y = random.randint(0, img_w - width), random.randint(0, img_h - height)  # 裁剪区域的起点

    img_croped = cv2.resize(img[y:y + height, x:x + width], (img_w, img_h))  # 裁减+调整回原图大小
    mask_croped = cv2.resize(mask[y:y + height, x:x + width], (img_w, img_h))  # 裁减+调整回原图大小
    notargets_croped = cv2.resize(notargets[y:y + height, x:x + width], (img_w, img_h))  # 裁减+调整回原图大小
    # print(img_croped.shape)
    # print(mask_croped.shape)
    # print(mask_croped.sum())
    # cv2.imshow('r2', mask_croped)
    # cv2.imshow('r1', img_croped)
    # cv2.waitKey(0)
    return img_croped, mask_croped, notargets_croped


# 镜像变换
def random_mirror_img_mask_notargets(img, mask, notargets):
    mode = random.randint(0, 1)  # mode = 1 水平翻转 mode = 0 垂直翻
    img_mirror = cv2.flip(img, mode)
    mask_mirror = cv2.flip(mask, mode)
    notargets_mirror = cv2.flip(notargets, mode)

    # print(img_mirror.shape)
    # print(mask_mirror.shape)
    # print(mask_mirror.sum())
    # cv2.imshow('r2', mask_mirror)
    # cv2.imshow('r1', img_mirror)
    # cv2.waitKey(0)
    return img_mirror, mask_mirror, notargets_mirror


# 仿射
def random_affine_img_mask_notargets(img, mask, notargets):
    img_h, img_w = img.shape[0], img.shape[1]
    mask_affined = np.zeros(1)
    point1 = np.float32([[2, 2], [3, 3.732], [4, 2]])
    while mask_affined.sum() == 0:
        x1 = random.uniform(0, 3)
        y1 = random.uniform(0, 3 - x1)
        x2 = random.uniform(0, 4)
        y2 = random.uniform(3.732, 4)
        x3 = random.uniform(3, 6)
        y3 = random.uniform(0, x3 - 3)
        point2 = np.float32([[x1, y1], [x2, y2], [x3, y3]])
        M = cv2.getAffineTransform(point1, point2)

        img_affined = cv2.warpAffine(img, M, (img_w, img_h))
        mask_affined = cv2.warpAffine(mask, M, (img_w, img_h))
        notargets_affined = cv2.warpAffine(notargets, M, (img_w, img_h))

        # print(img_affined.shape)
        # print(mask_affined.shape)
        # print(mask_affined.sum())
        # cv2.imshow('r2', mask_affined)
        # cv2.imshow('r1', img_affined)
        # cv2.waitKey(0)
    return img_affined, mask_affined, notargets_affined


