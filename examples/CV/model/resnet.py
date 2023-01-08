# This ResNet implementation is modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

import torch
import torch.nn as nn

class ResNetBottleNeck(nn.Module):

    expansion = 4

    def __init__(self, input_feature, feature, stride = 1):
        super(ResNetBottleNeck, self).__init__()

        # 1*1 conv
        self.conv1 = nn.Conv2d(input_feature, feature, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(feature)
        # 3*3 conv
        self.conv2 = nn.Conv2d(feature, feature, kernel_size = 3, stride = stride, padding = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(feature)
        # 1*1 conv
        self.conv3 = nn.Conv2d(feature, self.expansion * feature, kernel_size = 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * feature)

        self.shortcut = nn.Sequential()

        if stride != 1 or input_feature != self.expansion * feature:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_feature, self.expansion * feature, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * feature)
            )

    def forward(self, l):
        short_cut = l
        l = nn.functional.relu(self.bn1(self.conv1(l)))
        l = nn.functional.relu(self.bn2(self.conv2(l)))
        l = self.bn3(self.conv3(l))

        l = self.shortcut(short_cut) + l
        return nn.functional.relu(l)

class ResNetModel(nn.Module):
    def __init__(self, depth, group_func=ResNetBottleNeck, num_classes=100):
        super(ResNetModel, self).__init__()
        self.NUM_BLOCKS = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }

        self.input_features = 64
        self.num_blocks = self.NUM_BLOCKS[depth]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_channels = self.input_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.input_features),
            nn.ReLU(inplace=True))
        
        self.group_layer1 = self._make_layer(group_func, 64, self.num_blocks[0], 1)
        self.group_layer2 = self._make_layer(group_func, 128, self.num_blocks[1], 2)
        self.group_layer3 = self._make_layer(group_func, 256, self.num_blocks[2], 2)
        self.group_layer4 = self._make_layer(group_func, 512, self.num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Linear(512 * group_func.expansion, num_classes)

    def _make_layer(self, group_func, features, count, stride = 1):
        layers = []
        for i in range(count):
            current_stride = stride if i == 0 else 1
            layers.append(group_func(self.input_features, features, current_stride))
            self.input_features = features * group_func.expansion

        return nn.Sequential(*layers)

    def forward(self, l):
        l = self.conv1(l)

        l = self.group_layer1(l)
        l = self.group_layer2(l)
        l = self.group_layer3(l)
        l = self.group_layer4(l)
        l = torch.mean(l, dim=(2,3))
        l = self.fully_connected(l)

        return l
    
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)
