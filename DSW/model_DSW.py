import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from DSW.find_center_twovector import find_batch_centers
import torchvision


class DynamicShiftWindow(nn.Module):
    def __init__(self):

        super(DynamicShiftWindow, self).__init__()

        self.feature = torchvision.models.resnet34(num_classes=512)

        self.bn2 = nn.BatchNorm1d(512)
        self.act2 = nn.LeakyReLU(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 128, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.LeakyReLU(0.2)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 32, bias=False)

        resnet = torchvision.models.resnet34(pretrained=True)
        pretrained_dict = resnet.state_dict()
        model_dict = self.state_dict()

        pretrained_dict = {('feature.') + k: v for k, v in pretrained_dict.items() if ('feature.') + k in model_dict}
        pretrained_dict.pop('feature.fc.weight')
        pretrained_dict.pop('feature.fc.bias')

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.feature(x)

        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x = self.fc4(x).view(x.size(0), 2, -1)

        return x

    def train_dsw_loss(self, pred_vector, binary_map):
        assert pred_vector.shape[0] == binary_map.shape[0]
        batch_centers_vectors = find_batch_centers(binary_map)  # (64, 16, 16)
        tensor_batch_centers_vectors = torch.tensor(batch_centers_vectors, device=torch.device('cuda:0'),
                                                    dtype=torch.float32)
        loss = F.mse_loss(pred_vector, tensor_batch_centers_vectors)

        return loss


if __name__ == "__main__":
    model = DynamicShiftWindow()
    print(model)
    x = torch.rand(7, 3, 208, 208)
    model.cuda()
    y = model(x.cuda())
    print(y.shape)

    print(model.train_dsw_loss(y, np.array(torch.rand(7, 208, 208) * 225, np.uint8)))
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 208, 208), as_strings=True, print_per_layer_stat=True)
    print(flops, params)
