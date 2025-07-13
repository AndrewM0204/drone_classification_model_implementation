import torch
import torch.nn as nn
from sklearn.metrics import f1_score

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool = nn.AvgPool2d(3, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.skip_conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2)
        self.skip_conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        return self.bn(self.conv(self.avgpool(x))) + self.skip_conv2(self.skip_conv1(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, size=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels))
        for _ in range(size-1):
            self.block.append(nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1))
            self.block.append(nn.ReLU())
            self.block.append(nn.BatchNorm2d(in_channels))

    def forward(self, x):
        return self.block(x)

class HCCSA(nn.Module):
    def __init__(self, channels, height, width, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.channel_attention1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.LeakyReLU())
        self.channel_attention2 = nn.Sequential(
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid())
        
        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 5, padding=2),
            nn.LayerNorm([channels // reduction, height, width]),
            nn.LeakyReLU())
        self.spatial_attention2 = nn.Sequential(
            nn.Conv2d(channels // reduction, channels, 5, padding=2),
            nn.LayerNorm([channels, height, width]),
            nn.Sigmoid())
        
    def forward(self, x):
        sa1 = self.spatial_attention1(x)
        f3 = self.channel_attention2(self.channel_attention1(x) * sa1)
        f4 = self.spatial_attention2(sa1)
        return x * f3 * f4

class HCCSANet(nn.Module):
    def __init__(self, classes=8):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2))
        self.conv_block0 = ConvBlock(64, size=4)
        self.res_block1 = ResBlock(64, 128)
        self.conv_block1 = ConvBlock(128, size=2)
        self.res_block2 = ResBlock(128, 256)
        self.conv_block2 = ConvBlock(256, size=2)
        self.hccsa2 = HCCSA(256, 10, 10)
        self.res_block3 = ResBlock(256, 512)
        self.conv_block3 = ConvBlock(512, size=2)
        self.hccsa3 = HCCSA(512, 4, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Conv2d(512, 512, 1, stride=1),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0.5),
                                  nn.Conv2d(512, classes, 1, stride=1))
        
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.input_block(x)
        x = self.res_block1(self.conv_block0(x) + x)
        x = self.res_block2(self.conv_block1(x) + x)
        x = self.res_block3(self.conv_block2(x) + self.hccsa2(x) + x)
        return self.head(self.gap(self.conv_block3(x) + self.hccsa3(x) + x))

    def train_step(self,x_batch, y_batch):
        self.train()
        output = self.forward(x_batch)
        loss = self.loss_fn(output.squeeze(), y_batch.squeeze())
        return loss

    @torch.inference_mode()
    def val_step(self, x_batch, y_batch):
        self.eval()
        logits = self.forward(x_batch)
        loss = self.loss_fn(logits.squeeze(), y_batch.squeeze())
        pred = torch.argmax(logits.squeeze(), dim=-1)
        acc = f1_score(y_batch.cpu().numpy(), pred.cpu().numpy(), average='micro')
        return loss.item(), acc