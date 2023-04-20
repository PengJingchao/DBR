import os, time, cv2, sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CyclicLR, LambdaLR
import numpy as np
from tqdm import tqdm
import random
from tensorboardX import SummaryWriter

from models_denseformer import DenseFormer
from BR.utils import run_bath_images, prepare_model
from DSW.model_DSW import DynamicShiftWindow

from utils import *
from dataset.MFIRST import G1G2Dataset



def init_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def get_customized_schedule_with_warmup(optimizer, num_warmup_steps, d_model=1e-3, last_epoch=-1):  # d_model=1e-3
    def lr_lambda(current_step):
        current_step += 1

        armodel = current_step ** -0.5
        arg2 = current_step * (num_warmup_steps ** -1.5)
        return (d_model ** -0.5) * min(armodel, arg2)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(dsw_checkpoint, mae_checkpoint, head_checkpoint=None, RESUME=False):
    assert RESUME is False or head_checkpoint is not None, 'if RESUME, checkpoint must be specified!'
    # 输出
    root_result_dir = os.path.join('../outputs')
    os.makedirs(root_result_dir, exist_ok=True)
    model_result_dir = os.path.join(root_result_dir, 'models')
    os.makedirs(model_result_dir, exist_ok=True)
    images_dir = os.path.join(root_result_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # log
    log_dir = os.path.join(root_result_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir)

    # dataset
    trainsplit = G1G2Dataset(mode='train')
    trainset = DataLoader(trainsplit, batch_size=64, pin_memory=True, num_workers=4, shuffle=True, drop_last=True)
    testsplit = G1G2Dataset(mode='test')
    testset = DataLoader(testsplit, batch_size=1, pin_memory=True, num_workers=4, shuffle=False, drop_last=True)

    # Model
    model_mae_gan = prepare_model(mae_checkpoint, 'mae_vit_large_patch16')
    model_dsw = DynamicShiftWindow()
    dsw_checkpoint = torch.load(dsw_checkpoint)
    model_dsw.load_state_dict(dsw_checkpoint['model_state'])
    model_mae_gan.cuda()
    model_dsw.cuda()

    model_head = DenseFormer()
    if head_checkpoint is not None:
        head_checkpoint = torch.load(head_checkpoint)
        try:
            model_head.load_state_dict(head_checkpoint['model_state'])
        except:
            head_checkpoint['model'].pop('decoder_pred.weight')
            head_checkpoint['model'].pop('decoder_pred.bias')
            model_head.load_state_dict(head_checkpoint['model'], strict=False)
        print('Load .pth successfully....')
    model_head.cuda()

    # optimizer
    optimizer = torch.optim.AdamW(model_head.parameters(), lr=1e-1, betas=(0.5, 0.999), weight_decay=5e-4)
    if RESUME:
        optimizer.load_state_dict(head_checkpoint['optimizer_state'])
        print('Resume last training....')

    scheduler_model_warmup = get_customized_schedule_with_warmup(optimizer, num_warmup_steps=1000, d_model=728)  # num_warmup_steps=200
    scheduler_model_pred = CyclicLR(optimizer, base_lr=1e-2, max_lr=5e-2, step_size_up=15, step_size_down=15,
                                 cycle_momentum=False)
    scheduler_model_all = CyclicLR(optimizer, base_lr=3e-5, max_lr=2e-4, step_size_up=15, step_size_down=15,
                                cycle_momentum=False)

    iteration = 0
    start_epoch = 0
    if RESUME:
        start_epoch = head_checkpoint['epoch']
        iteration = head_checkpoint['iteration']
        scheduler_model_all.step()

    for epoch in range(start_epoch, start_epoch + 1000):
        if epoch < 181:  # TODO 100*1
            model_head.freeze_part()
        else:
            model_head.unfreeze_part()

        for bt_idx, data in enumerate(tqdm(trainset)):
            torch.cuda.empty_cache()  
            iteration = iteration + 1
            summary_writer.add_scalar('lr', float(optimizer.param_groups[0]['lr']), iteration)

            input_images, output_images = data['input_images'], data['output_images']
            input_images = input_images.cuda(non_blocking=True).float()
            output_images = output_images.cuda(non_blocking=True).float()

            model_dsw.eval()
            model_mae_gan.eval()
            model_head.train()
            optimizer.zero_grad()

            with torch.no_grad():
                swift_offset = model_dsw(input_images)
                no_target = run_bath_images(torch.argmax(swift_offset, dim=2), input_images, model_mae_gan)
                input_images = torch.einsum('j,ijkl', torch.tensor((0.299, 0.387, 0.114)).cuda(), input_images)
                no_target = torch.einsum('j,ijkl', torch.tensor((0.299, 0.387, 0.114)).cuda(), no_target)
                input_head = torch.stack((input_images, no_target, input_images - no_target), dim=1)

            model_out = model_head(input_head)
            model_out = torch.clamp(model_out, 0.0, 1.0)

            axes = tuple(range(1, len(model_out.shape)))
            
            gen_loss1 = -1 * torch.log(torch.mean(0.002 * 2 * torch.sum(model_out * output_images, axes) / (torch.sum(model_out, axes) + 0.002 * torch.sum(output_images, axes) + 1e-6)))

            summary_writer.add_scalar('trainloss', gen_loss1, iteration)

            gen_loss1.backward()
            optimizer.step()
            if epoch in range(181, 381):
                scheduler_model_warmup.step()

        # test
        sum_val_loss_model = 0
        sum_val_pr_model = 0
        sum_val_re_model = 0
        sum_val_F1_model = 0

        for bt_idx_test, data in enumerate(tqdm(testset)):
            model_head.eval()
            model_dsw.eval()
            model_mae_gan.eval()
            optimizer.zero_grad()

            with torch.no_grad():
                input_images, output_images = data['input_images'], data['output_images']  # [B, 1, 128, 128]
                input_images = input_images.cuda(non_blocking=True).float()
                output_images = output_images.cuda(non_blocking=True).float()

                swift_offset = model_dsw(input_images)
                no_target = run_bath_images(torch.argmax(swift_offset, dim=2), input_images, model_mae_gan)
                input_images = torch.einsum('j,ijkl', torch.tensor((0.299, 0.387, 0.114)).cuda(), input_images)
                no_target = torch.einsum('j,ijkl', torch.tensor((0.299, 0.387, 0.114)).cuda(), no_target)
                input_head = torch.stack((input_images, no_target, input_images - no_target), dim=1)

                model_out = model_head(input_head)  # [B, 1, 128, 128]
                model_out = torch.clamp(model_out, 0.0, 1.0)

                output_images = output_images.cpu().numpy() / 255.
                model_out = model_out.detach().cpu().numpy()

                val_loss_model = np.mean(np.square(model_out - output_images))
                sum_val_loss_model += val_loss_model
                precision, recall, val_F1_model, _ = calculatePreRecF1Measure(model_out, output_images, 0.5)
                sum_val_pr_model += precision
                sum_val_re_model += recall
                sum_val_F1_model += val_F1_model


        # logger.info("======================== model_head results ============================")
        avg_val_loss_model = sum_val_loss_model / len(testsplit)
        avg_val_pr_model = sum_val_pr_model / len(testsplit)
        avg_val_re_model = sum_val_re_model / len(testsplit)
        avg_val_F1_model = sum_val_F1_model / len(testsplit)

        summary_writer.add_scalar('valloss', avg_val_loss_model, epoch + 1)
        summary_writer.add_scalar('Pr_rate', avg_val_pr_model, epoch + 1)
        summary_writer.add_scalar('Re_rate', avg_val_re_model, epoch + 1)
        summary_writer.add_scalar('F1_score', avg_val_F1_model, epoch + 1)

        print('current epoch {}/{}, total iteration: {}, Pr: {}, Re: {}, F1: {}'.format(
                        epoch + 1, 1000, iteration, avg_val_pr_model, avg_val_re_model, avg_val_F1_model))

        ############# save model_head
        ckpt_name = os.path.join(model_result_dir, 'epoch_{}_batch_{}.pth'.format(epoch + 1, bt_idx + 1))
        state = {'model_state': model_head.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch,
                 'iteration': iteration}
        torch.save(state, ckpt_name)

        if epoch < 181:  # TODO 100*3
            scheduler_model_pred.step()
        elif epoch >= 381:  # TODO 200*3
            scheduler_model_all.step()


if __name__ == '__main__':
    mae_checkpoint = '<path to your pre-trained .pth>'
    dsw_checkpoint = '<path to your pre-trained .pth>'
    head_checkpoint = '.<path to your pre-trained .pth>'
    train(dsw_checkpoint, mae_checkpoint, head_checkpoint, RESUME=False)
    
    
    
