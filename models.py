import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5) # 32 x 220 x 220
        self.pool1 = nn.MaxPool2d(2, 2) # 32 x 110 x 110
        self.batch_norm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3) # 64 x 108 x 108
        self.pool2 = nn.MaxPool2d(2, 2) # 64 x 54 x 54
        self.batch_norm2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 54 * 54, 256)
        self.fc1_dropout = nn.Dropout(p=0.25)
        self.fc1_batchnorm = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 2*68)

        I.uniform_(self.conv1.weight)
        I.uniform_(self.conv2.weight)
        I.xavier_uniform_(self.fc1.weight)
        I.xavier_uniform_(self.fc2.weight)

        
    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.batch_norm1(x)
        
        x = self.conv2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.batch_norm2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc1_dropout(x)
        x = self.fc1_batchnorm(x)
        
        x = self.fc2(x)
        
        return x
