import torch
import torch.nn as nn
import torch.nn.functional as F

bn_momentum = 0.1

class ResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class FeatureNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size = 3, padding = 1, stride=2),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True))

        self.forw1 = nn.Sequential(
            ResBlock3d(24, 32),
            ResBlock3d(32, 32))

        self.forw2 = nn.Sequential(
            ResBlock3d(32, 64),
            ResBlock3d(64, 64))

        self.forw3 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        self.forw4 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        # skip connection in U-net
        self.back2 = nn.Sequential(
            # 64 + 64 + 3, where 3 is the channeld dimension of coord
            ResBlock3d(128, 128),
            ResBlock3d(128, 128),
            ResBlock3d(128, 128))

        # skip connection in U-net
        self.back3 = nn.Sequential(
            ResBlock3d(128, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)

        # upsampling in U-net
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        # upsampling in U-net
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        
        self.gap=nn.AdaptiveAvgPool3d((None,1,1))#global average pooling
        
        self.fc1=nn.Linear(128*16, 128)##new
        self.fc2=nn.Linear(128, out_channels)##new 
    
    def forward(self, x):
        out = self.preBlock(x)#16
        out_pool = out
        
        out1 = self.forw1(out_pool)#32
        out1_pool, _ = self.maxpool2(out1)
        
        out2 = self.forw2(out1_pool)#64
        out2_pool, _ = self.maxpool3(out2)
        
        out3 = self.forw3(out2_pool)#96
        out3_pool, _ = self.maxpool4(out3)
        
        out4 = self.forw4(out3_pool)#96

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))#96+96
        
        rev2 = self.path2(comb3)
        comb2 = self.back2(torch.cat((rev2, out2), 1))#64+64
        
        gap = self.gap(comb2)
        
        x = gap.view(-1, 128*16)
        x = F.relu(self.fc1(x), inplace=True)
        out = self.fc2(x)
        
        return out