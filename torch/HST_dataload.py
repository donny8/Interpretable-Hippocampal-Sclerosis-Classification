from HST_common import *


class BasicLinear(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels, bias=True)
        self.drop = nn.Dropout(p=drop_rate)
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.drop(x)
        return x
    
class Basic3D(nn.Module):

    def __init__(self,num_classes=classes):
        super().__init__()

        NumFilter = [5,10,20,40,60] # Number of Convolutional Filters to use
        kernel_size = [3,3,3] # Convolution Kernel Size
        stride_size = (1,1,1) # Convolution Stride Size
        pool_size = [2,2,2]
        zero_pad = [1,1,1] # Convolution Zero Padding Size

        self.conv3_1 = nn.Conv3d(1, NumFilter[0], bias=True, kernel_size=kernel_size, stride=stride_size, padding = zero_pad)
        self.bn_1 = nn.BatchNorm3d(NumFilter[0])
        self.conv3_2 = nn.Conv3d(NumFilter[0],NumFilter[1], bias=True, kernel_size=kernel_size, stride=stride_size, padding = zero_pad)
        self.bn_2 = nn.BatchNorm3d(NumFilter[1])
        self.conv3_3 = nn.Conv3d(NumFilter[1],NumFilter[2], bias=True, kernel_size=kernel_size, stride=stride_size, padding = zero_pad)
        self.bn_3 = nn.BatchNorm3d(NumFilter[2])
        self.classifier = nn.Sequential(
            BasicLinear(210000,32),
            BasicLinear(32,8),
            nn.Linear(8,num_classes),
            nn.Softmax(dim=0)
        )
        
    def forward(self, x):
        x = self.conv3_1(x)
        x = F.relu(F.max_pool3d(self.bn_1(x),kernel_size=pool_size))
        x = self.conv3_2(x)
        x = F.relu(F.max_pool3d(self.bn_2(x),kernel_size=pool_size))
        x = self.conv3_3(x)
        x = F.relu(F.max_pool3d(self.bn_3(x),kernel_size=pool_size))
        x = torch.flatten(x, 0)
        x = x.view(-1,210000)
        x = self.classifier(x)
        return x

def HSCNN():
    return Basic3D()