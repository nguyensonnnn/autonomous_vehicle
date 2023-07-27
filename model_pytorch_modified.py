import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(1152, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)
              
        
        
    
    def forward(self, x):
        """Forward pass."""
        x = x.reshape(x.size(0), 3, 66, 200)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.dropout(x,p=0.5)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
       
        x = F.elu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
       
        x = F.elu(self.fc3(x))
        x = F.dropout(x, p=0.5, training=self.training)
       
        x = F.elu(self.fc4(x))
        x = F.dropout(x, p=0.5, training=self.training)
       
        x = torch.atan(self.fc5(x)) * 2.0  # Scale the atan output
        return x

        


# Create an instance of the model
model = MyModel()

# Create a random input tensor of size (batch_size,  height, width,channels,)
x = torch.empty((50, 3, 66, 200),dtype=torch.float32)

# Pass the input through the model
output = model(x)
