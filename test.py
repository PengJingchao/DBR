import cv2
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Resize

from dataset.MFIRST import G1G2Dataset
# from dataset.SIRST import G1G2Dataset
from utils import *

sys.path.append('BR')
sys.path.append('DSW')
sys.path.append('DH')
from BR.utils import run_one_image_for_test, prepare_model
from DSW.model_DSW import DynamicShiftWindow
from DH.models_denseformer import DenseFormer


def test(dsw_checkpoint, mae_checkpoint, head_checkpoint, save_pic=False):
    # 输出的总路径
    root_result_dir = os.path.join('outputs')
    os.makedirs(root_result_dir, exist_ok=True)
    image_result_dir = os.path.join(root_result_dir, 'images')
    os.makedirs(image_result_dir, exist_ok=True)

    # dataset
    testsplit = G1G2Dataset(mode='test')
    testset = DataLoader(testsplit, batch_size=1, pin_memory=True,
                         num_workers=4, shuffle=False, drop_last=True)

    # 3 Model
    model_mae_gan = prepare_model(mae_checkpoint, 'mae_vit_large_patch16')
    model_dsw = DynamicShiftWindow()
    model_head = DenseFormer()
    dsw_checkpoint = torch.load(dsw_checkpoint)
    head_checkpoint = torch.load(head_checkpoint)
    model_dsw.load_state_dict(dsw_checkpoint['model_state'])
    model_head.load_state_dict(head_checkpoint['model_state'])
    print('Load checkpoint successfully....')
    model_mae_gan.cuda()
    model_dsw.cuda()
    model_head.cuda()

    # test
    sum_val_loss = 0
    sum_val_false_ratio = 0
    sum_val_detect_ratio = 0
    sum_val_Precision = 0
    sum_val_Recall = 0
    sum_val_F1 = 0
    sum_val_IOU = 0
    g1_time = 0

    torch_resize = Resize([224, 224])

    for bt_idx_test, data in enumerate(tqdm(testset)):
        model_mae_gan.eval()
        model_dsw.eval()
        model_head.eval()

        with torch.no_grad():
            
            input_images, masks = data['input_images'], data['output_images']
            input_images = input_images.cuda(non_blocking=True).float()
            masks = masks.cuda(non_blocking=True).float()

            stime = time.time()
            y = model_dsw(input_images)
            no_target = run_one_image_for_test(torch.argmax(y, dim=2)[0][0], torch.argmax(y, dim=2)[0][1], input_images,
                                               model_mae_gan)
            input_images = torch.einsum('j,ijkl', torch.tensor((0.299, 0.387, 0.114)).cuda(), input_images)
            no_target = torch.einsum('j,ijkl', torch.tensor((0.299, 0.387, 0.114)).cuda(), no_target)
            input_head = torch.stack((input_images, no_target, input_images - no_target), dim=1)

            model_out = model_head(input_head)
            model_out = torch.clamp(model_out, 0.0, 1.0)

            etime = time.time()
            g1_time += etime - stime

            masks = masks.cpu().numpy()
            model_out = model_out.detach().cpu().numpy()
            val_loss_g1 = np.mean(np.square(model_out - masks))
            sum_val_loss += val_loss_g1
            val_false_ratio_g1 = np.mean(np.maximum(0, model_out - masks))
            sum_val_false_ratio += val_false_ratio_g1
            val_detect_ratio_g1 = np.sum(model_out * masks) / np.maximum(np.sum(masks), 1)
            sum_val_detect_ratio += val_detect_ratio_g1
            val_Precision_g1, val_Recall_g1, val_F1_g1, val_IOU_g1 = calculatePreRecF1Measure(model_out, masks / 255., 0.5)
            sum_val_Precision += val_Precision_g1
            sum_val_Recall += val_Recall_g1
            sum_val_F1 += val_F1_g1
            sum_val_IOU += val_IOU_g1

        # 保存图片
        if save_pic:
            output_image1 = np.squeeze(model_out * 255.0)  # /np.maximum(output_image1.max(),0.0001))
            cv2.imwrite(os.path.join(image_result_dir, "%05d.png" % bt_idx_test), np.uint8(output_image1))

    print("======================== g1 results ============================")
    avg_val_loss_g1 = sum_val_loss / len(testsplit)
    avg_val_false_ratio_g1 = sum_val_false_ratio / len(testsplit)
    avg_val_detect_ratio_g1 = sum_val_detect_ratio / len(testsplit)
    avg_val_Presicion_g1 = sum_val_Precision / len(testsplit)
    avg_val_Recall_g1 = sum_val_Recall / len(testsplit)
    avg_val_F1_g1 = sum_val_F1 / len(testsplit)
    avg_val_IOU_g1 = sum_val_IOU / len(testsplit)

    print("================val_L2_loss is %f" % avg_val_loss_g1)
    print("================falseAlarm_rate is %f" % avg_val_false_ratio_g1)
    print("================detection_rate is %f" % avg_val_detect_ratio_g1)
    print("================Presicion measure is %f" % avg_val_Presicion_g1)
    print("================Recall measure is %f" % avg_val_Recall_g1)
    print("================F1 measure is %f" % avg_val_F1_g1)
    print("================IOU measure is %f" % avg_val_IOU_g1)
    print("g1 time is {}".format(g1_time))
    return avg_val_F1_g1


if __name__ == '__main__':
    mae_checkpoint = '<path to BR-.pth>'
    dsw_checkpoint = '<path to your DSW>'
    head_checkpoint = '<path to your DH>'
    test(dsw_checkpoint, mae_checkpoint, head_checkpoint, save_pic=True)
    
