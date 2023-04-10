import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.abspath('.'))
import BR.MAE

sys.path.append('..')

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


# imagenet_mean = np.array([104.98419416/255, 104.98059531/255, 104.99451429/255])
# imagenet_std = np.array([26.87694834/255, 26.87644202/255, 26.87816581/255])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def show_subtracted_image(image1, image2, title=''):
    # image is [H, W, 3]
    assert image1.shape[2] == 3
    plt.imshow(torch.clip(((image1 * imagenet_std + imagenet_mean) - (image2 * imagenet_std + imagenet_mean)) * 255, 0,
                          255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(BR.MAE, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(center_x, center_y, img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    delta_x = (24 - center_x) % 16  # belongs to [0,16)
    delta_y = (24 - center_y) % 16  # belongs to [0,16)
    assert 0 <= delta_y < 16
    assert 0 <= delta_x < 16

    '''cat to get a large pic'''
    x_large = torch.cat([x[:, :, :, :], x[:, :, 12 * 16:, :]], 2)
    x_large = torch.cat([x_large[:, :, :, :], x_large[:, :, :, 12 * 16:]], 3)

    # swift window
    x_large = torch.roll(x_large, shifts=(delta_y, delta_x), dims=(2, 3))

    # run MAE
    loss_large, y_large, mask_large = model(x_large.float().cuda(), mask_ratio=0.50)
    y_large = model.unpatchify(y_large)
    # visualize the mask
    mask_large = mask_large.detach()
    mask_large = mask_large.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask_large = model.unpatchify(mask_large)  # 1 is removing, 0 is keeping

    # reverse shift window
    # x_large = x_large[:,:,delta_y:208+delta_y,delta_x:208+delta_x]
    y_large = y_large[:, :, delta_y:208 + delta_y, delta_x:208 + delta_x]
    mask_large = mask_large[:, :, delta_y:208 + delta_y, delta_x:208 + delta_x]

    # nchw->nhwcTo visualize
    x = torch.einsum('nchw->nhwc', x)
    y_large = torch.einsum('nchw->nhwc', y_large).detach().cpu()
    mask_large = torch.einsum('nchw->nhwc', mask_large).detach().cpu()

    '''Merge'''
    # x = x_large
    y = y_large
    mask = mask_large

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    # plt.subplot(1, 4, 4)
    # show_image(im_paste[0], "reconstruction + visible")

    plt.subplot(1, 4, 4)
    show_subtracted_image(x[0], y[0], "original - reconstruction")

    plt.show()


def run_one_image_for_test(center_x, center_y, img, model):
    x = img

    delta_x = (24 - center_x) % 16  # belongs to [0,16)
    delta_y = (24 - center_y) % 16  # belongs to [0,16)
    assert 0 <= delta_y < 16
    assert 0 <= delta_x < 16

    '''cat to get a large pic'''
    x_large = torch.cat([x[:, :, :, :], x[:, :, 12 * 16:, :]], 2)
    x_large = torch.cat([x_large[:, :, :, :], x_large[:, :, :, 12 * 16:]], 3)

    # swift window
    x_large = torch.roll(x_large, shifts=(delta_y, delta_x), dims=(2, 3))

    # run MAE
    _, y_large, _ = model(x_large.float().cuda(), mask_ratio=0.50)
    y_large = model.unpatchify(y_large)

    # reverse shift window
    y_large = y_large[:, :, delta_y:208 + delta_y, delta_x:208 + delta_x]

    return y_large


def run_bath_images(center, img, model):
    center = (24 - center) % 16  # belongs to [0,16)
    # cat to get a large pic
    x_large_batch = torch.cat([img[:, :, :, :], img[:, :, 12 * 16:, :]], 2)
    x_large_batch = torch.cat([x_large_batch[:, :, :, :], x_large_batch[:, :, :, 12 * 16:]], 3)
    y = torch.zeros_like(img)

    # swift window
    for x_index in range(img.shape[0]):
        x_large_batch[x_index, :, :, :] = torch.roll(x_large_batch[x_index, :, :, :],
                                                     shifts=(center[x_index, 1], center[x_index, 0]), dims=(1, 2))

    # run MAE
    _, y_large, _ = model(x_large_batch.float().cuda(), mask_ratio=0.50)
    y_large = model.unpatchify(y_large)

    # reverse shift window
    for y_index in range(img.shape[0]):
        y[y_index, :, :, :] = y_large[y_index, :, center[y_index, 1]:208 + center[y_index, 1],
                                      center[y_index, 0]:208 + center[y_index, 0]]

    return y


if __name__ == '__main__':
    # load an image
    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'  # fox, from ILSVRC2012_val_00046145
    # img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
    # img = Image.open(requests.get(img_url, stream=True).raw)
    img = Image.open('/home/pjc/MyProgram/2022/BRT/data/MFIRST/test_org/00036.png')
    img = img.resize((208, 208))
    img = np.array(img) / 255.

    # assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    plt.rcParams['figure.figsize'] = [5, 5]
    show_image(torch.tensor(img))

    # chkpt_dir = './mae_visualize_vit_large.pth'
    # model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    # print('Model loaded.')
    #
    # # make random mask reproducible (comment out to make it change)
    # torch.manual_seed(2)
    # print('MAE with pixel reconstruction:')
    # model_mae.cuda()
    # run_one_image(img, model_mae)

    # This is an MAE model trained with an extra GAN loss for more realistic generation (ViT-Large, training mask ratio=0.75)
    # download checkpoint if not exist
    # !wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth

    chkpt_dir = './mae_visualize_vit_large_ganloss.pth'
    model_mae_gan = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print('MAE with extra GAN loss:')
    model_mae_gan.cuda()
    run_one_image(12, 3, img, model_mae_gan)
