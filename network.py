# encoding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import BaseBlock

class MobileNetV2(nn.Module):
    def __init__(self, output_size, alpha = 1):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        # first conv layer

        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks1 = nn.Sequential(
            BaseBlock(32, 16, t = 1, downsample = False),


            )
        self.bottlenecks2 = nn.Sequential(
            BaseBlock(16, 24, downsample=False)
        )
        self.bottlenecks3 = nn.Sequential(
            BaseBlock(24, 24)
        )
        self.bottlenecks4 = nn.Sequential(
            BaseBlock(24, 32, downsample = False)
        )
        self.bottlenecks5 = nn.Sequential(
            BaseBlock(32, 32)
        )
        self.bottlenecks6 = nn.Sequential(
            BaseBlock(32, 32)
        )

        self.bottlenecks7 = nn.Sequential(
            BaseBlock(32, 64, downsample = True)
        )

        self.bottlenecks8 = nn.Sequential(
            BaseBlock(64, 64),



        )
        self.bottlenecks9 = nn.Sequential(
            BaseBlock(64, 64)
        )
        self.bottlenecks10 = nn.Sequential(
            BaseBlock(64, 64)
        )
        self.bottlenecks11 = nn.Sequential(
            BaseBlock(64, 96, downsample=False)
        )
        self.bottlenecks12 = nn.Sequential(
            BaseBlock(96, 96)
        )
        self.bottlenecks13 = nn.Sequential(
            BaseBlock(96, 96)
        )
        self.bottlenecks14 = nn.Sequential(
            BaseBlock(96, 160, downsample=True)
        )
        self.bottlenecks15 = nn.Sequential(
            BaseBlock(160, 160)
        )

        self.bottlenecks16 = nn.Sequential(
            BaseBlock(160, 160)
        )
        self.bottlenecks17 = nn.Sequential(
            BaseBlock(160, 320, downsample=False)
        )
        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size)

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward1(self,inputs):
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace=True)

        return x
    def forward2(self,inputs):
        x = self.bottlenecks1(inputs)
        return x

    def forward3(self,inputs):
        x = self.bottlenecks2(inputs)
        return x

    def forward4(self,inputs):
        x = self.bottlenecks3(inputs)
        return x

    def forward5(self,inputs):
        x = self.bottlenecks4(inputs)
        return x

    def forward6(self,inputs):
        x = self.bottlenecks5(inputs)
        return x

    def forward7(self,inputs):
        x = self.bottlenecks6(inputs)
        return x

    def forward8(self,inputs):
        x = self.bottlenecks7(inputs)
        return x

    def forward9(self,inputs):
        x = self.bottlenecks8(inputs)
        return x

    def forward10(self,inputs):
        x = self.bottlenecks9(inputs)
        return x

    def forward11(self,inputs):
        x = self.bottlenecks10(inputs)
        return x

    def forward12(self,inputs):
        x = self.bottlenecks11(inputs)
        return x

    def forward13(self,inputs):
        x = self.bottlenecks12(inputs)
        return x

    def forward14(self,inputs):
        x = self.bottlenecks13(inputs)
        return x

    def forward15(self,inputs):
        x = self.bottlenecks14(inputs)
        return x

    def forward16(self,inputs):
        x = self.bottlenecks15(inputs)
        return x

    def forward17(self,inputs):
        x = self.bottlenecks16(inputs)
        return x

    def forward18(self,inputs):
        x = self.bottlenecks17(inputs)
        return x


    def forward19(self,inputs):
        x = F.relu6(self.bn1(self.conv1(inputs)), inplace = True)
        return x

    def forward20(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

