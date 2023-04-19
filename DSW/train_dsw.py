import torch, os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CyclicLR, LambdaLR
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import random

from dataset.MFIRST import G1G2Dataset
# from dataset.SIRST import G1G2Dataset
from config import config

from model_DSW import DynamicShiftWindow

import sys

sys.path.append('../dataset')


def init_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def train(checkpoint_path=None, RESUME=False):
    # 保存输出的总路径
    root_result_dir = os.path.join('../outputs')
    os.makedirs(root_result_dir, exist_ok=True)
    model_result_dir = os.path.join(root_result_dir, 'models')
    os.makedirs(model_result_dir, exist_ok=True)

    # 日志文件
    log_dir = os.path.join(root_result_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir)

    # 定义dataset
    trainsplit = G1G2Dataset(mode='train')
    trainset = DataLoader(trainsplit, batch_size=config.mini_batch_size, pin_memory=True,
                          num_workers=4, shuffle=True, drop_last=True)
    testsplit = G1G2Dataset(mode='test')
    testset = DataLoader(testsplit, batch_size=1, pin_memory=True,
                         num_workers=4, shuffle=False, drop_last=True)

    # 定义model
    model = DynamicShiftWindow()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
    if RESUME:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    model.cuda()

    scheduler = CyclicLR(optimizer, base_lr=config.base_lr, max_lr=config.max_lr, step_size_up=config.step_size_up,
                         step_size_down=config.step_size_down, cycle_momentum=False)

    iteration = 0
    start_epoch = 0
    if RESUME:
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']

    for epoch in range(start_epoch, config.max_epoch_num + start_epoch):
        # 训练一个周期
        model.train()
        if hasattr(model, 'feature'):
            print('freeze feature extraction module')
            for k in model.feature.parameters():
                k.requires_grad = False
            try:
                model.feature.fc.weight.requires_grad = True
                model.feature.fc.bias.requires_grad = True
            except:
                model.feature.classifier[6].weight.requires_grad = True
                model.feature.classifier[6].bias.requires_grad = True
        for bt_idx, data in enumerate(tqdm(trainset)):
            # 训练一个batch
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            iteration = iteration + 1
            summary_writer.add_scalar('lr', float(optimizer.param_groups[0]['lr']), iteration)

            input_images, output_images = data['input_images'], data['output_images']
            input_images = input_images.cuda(non_blocking=True).float()
            # output_images = output_images.cuda(non_blocking=True).float()

            y = model(input_images)
            loss = model.train_dsw_loss(y, np.array(output_images, np.uint8))

            summary_writer.add_scalar('loss', loss, iteration)
            loss.backward()
            optimizer.step()

        val_loss = 0
        model.eval()
        for bt_idx_test, data_test in enumerate(tqdm(testset)):
            optimizer.zero_grad()
            input_images_test, output_images_test = data_test['input_images'], data_test['output_images']
            input_images_test = input_images_test.cuda(non_blocking=True).float()
            with torch.no_grad():
                y = model(input_images_test)
                print(torch.argmax(y, dim=-1))
                loss = model.train_dsw_loss(y, np.array(output_images_test, np.uint8))
                val_loss = val_loss + loss
                print(loss)

        summary_writer.add_scalar('valloss', val_loss / len(testset), epoch + 1)

        ckpt_name = os.path.join(model_result_dir, 'epoch_{}_batch_{}.pth'.format(epoch + 1, bt_idx + 1))

        print("Epoch %d valid_loss: %.4f lr: %e" % (
            epoch, val_loss / len(testset), float(optimizer.param_groups[0]['lr'])))
        state = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch,
                 'iteration': iteration}
        torch.save(state, ckpt_name)
        scheduler.step()


if __name__ == '__main__':
    init_seed(56438)
    train()
