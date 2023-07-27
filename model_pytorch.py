import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Dropout(0.5)
        )
        
        #self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear_layers= nn.Sequential(
            nn.Linear(in_features=64*1*18, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1),
        )
        
                
    
    def forward(self, x):
        """Forward pass."""
        x = x.reshape(x.size(0), 3, 66, 200)
        output=self.conv_layers(x)
        output = output.reshape(x.size(0), -1)  # Flatten the tensor
        output=self.linear_layers(output)
        #x = torch.atan(self.fc5(x)) * 2.0  # Scale the atan output
        return output


# Create an instance of the model
model = MyModel()

# Create a random input tensor of size (batch_size,  height, width,channels,)
x = torch.empty((50, 3, 66, 200),dtype=torch.float32)

# Pass the input through the model
output = model(x)