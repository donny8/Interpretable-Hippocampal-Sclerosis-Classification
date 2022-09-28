from .common import *

    

class Torch3D(nn.Module):

    def __init__(self,ksize,num_classes=nb_classes):
        super().__init__()
        self.ksize = ksize
        if(self.ksize==3): self.linear=48000
        elif(self.ksize==4): self.linear=35640

        NumFilter = [5,10,20,40,60] # Number of Convolutional Filters to use
        NumDense = [64,64,3]
        kernel_size = [self.ksize,self.ksize,self.ksize] # Convolution Kernel Size
        stride_size = (1,1,1) # Convolution Stride Size
        pad_size = [1,1,1] # Convolution Zero Padding Size
        pool_size = [2,2,2]
        
        self.extractor = nn.Sequential(
            nn.Conv3d(1, NumFilter[0], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(NumFilter[0],NumFilter[1], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(NumFilter[1],NumFilter[2], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(NumFilter[2],NumFilter[3], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.linear, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),

            nn.Linear(64, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),

            nn.Linear(64, num_classes, bias=True)
            )
        
    def forward(self, x):
        x = self.extractor(x)
        x = x.view(-1,self.linear)
        x = self.classifier(x)
        
        return x

def HSCNN(ksize):
    return Torch3D(ksize)
