import cv2
import os

import numpy as np
from torch.utils.data.dataset import Dataset
from dataset.data_augmentation import *

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


# mean = [104.98419416, 104.98059531, 104.99451429]
# std = [26.87694834, 26.87644202, 26.87816581]


class G1G2Dataset(Dataset):
    def __init__(self, mode):
        self.mode = mode

        if self.mode == 'train':
            self.imageset_dir = os.path.join('<path to your MFIRST>/training/')
            self.imageset_gt_dir = os.path.join('<path to your MFIRST>/training/')
        elif self.mode == 'test':
            self.imageset_dir = os.path.join('<path to your MFIRST>/test_org/')
            self.imageset_gt_dir = os.path.join('<path to your MFIRST>/test_gt/')
        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode == 'train':
            return 9900
        elif self.mode == 'test':
            return 100
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_dir = os.path.join(self.imageset_dir, "%06d_1.png" % idx)
            gt_dir = os.path.join(self.imageset_gt_dir, "%06d_2.png" % idx)
        elif self.mode == 'test':
            img_dir = os.path.join(self.imageset_dir, "%05d.png" % idx)
            gt_dir = os.path.join(self.imageset_gt_dir, "%05d.png" % idx)
        else:
            raise NotImplementedError

        img = cv2.imread(img_dir, 1)
        mask = cv2.imread(gt_dir, 0)

        if self.mode == 'train':
            mode = random.randint(0, 5)
            if mode % 5 == 0:
                img, mask = random_rotate_img_mask(img, mask)
            if mode % 5 == 1:
                img, mask = random_crop_img_mask(img, mask)
            if mode % 5 == 2:
                img, mask = random_mirror_img_mask(img, mask)
            if mode % 5 == 3:
                img, mask = random_affine_img_mask(img, mask)

        img = np.float32(img) / 255.
        if img is None:
            print('Could not open or find the image:', img_dir)
            exit(0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        real_input = (img - imagenet_mean) / imagenet_std
        real_input = cv2.resize(real_input, (208, 208))

        input_images = np.swapaxes(real_input, 1, 2)
        input_images = np.swapaxes(input_images, 0, 1)

        mask = cv2.resize(mask, (208, 208))
        # mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST_EXACT)
        # dilated_mask = mask
        # output_images = np.float32(dilated_mask) / 255.0  # 像素归一化
        # output_images = np.expand_dims(output_images, axis=0)
        output_images = mask

        sample_info = {}
        sample_info['input_images'] = input_images
        sample_info['output_images'] = output_images

        return sample_info


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import os

    os.chdir('../')

    trainsplit = G1G2Dataset(mode='train')
    trainset = DataLoader(trainsplit, batch_size=64, pin_memory=True, num_workers=4, shuffle=False, drop_last=True)
    for data in trainset:
        input_images, output_images = data['input_images'], data['output_images']
        print(input_images.shape, output_images.shape)
