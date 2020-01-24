## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # maxpool that uses a square window of kernel_size=2, stride=2
        # Defining the Covolutional Layers, maxpooling layers and dropouts
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) 

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3) 

        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3)

        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3) 
        
        # Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Defining Three Fully Connected Linear Layers 
        self.fc1 = nn.Linear(in_features = 512*5*5, out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 512)
        # the output 136 in order to having 2 for each of the 68 keypoint (x, y) pairs
        self.fc3 = nn.Linear(in_features = 512, out_features = 136)

        # Dropouts
        self.dropout = nn.Dropout(p = 0.3)

       
    # Define the feedforward behavior    
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))
        
        # Convolution + Activation + Pooling     
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Flattening the feature maps into feature vectors
        #x = x.view(x.size(0), -1)
        x = x.view(-1, self.num_flat_features(x))  
          
        # Fully connected Linear layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) 
        
        # final output
        return x
    
   
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features