import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize
import rasterio
import pandas as pd
from typing import Optional, List

# define class that will contain all the image tiles and matching labels 
# and serve as input to the deep learning model
# this version is adapted for use with tif-files
class TreeSpeciesClassificationDataset(Dataset):
    def __init__(self, img_data: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.img_data: pd.DataFrame = img_data
        self.transform: Optional[transforms.Compose] = transform
        
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        img_path = self.img_data.loc[index, "images"]  # TODO document that this expects that data is either preprocessed with this module or there exists a column called 'images' with the absolute path to the image
        img_label = torch.tensor(self.img_data.loc[index, 'encoded_labels'])  # TODO see above TODO
        with rasterio.open(img_path) as f:
            image_np = f.read()
        image_np = image_np[:3, :, :]
        image_tensor = ToTensor()(image_np)
        image_tensor = image_tensor.permute(1,2,0)
        image_tensor = Resize((100, 100))(image_tensor)
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        return image_tensor, img_label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 3 input image channel, 16 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16, 32,kernel_size=3,stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64,kernel_size=3,stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64,kernel_size=3,stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        #self.dropout = nn.Dropout2d(0.4)
        self.dropout = nn.Dropout(0.4)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*2*2,512 )
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)
        
    def forward(self, x):
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.dropout(self.batchnorm2(self.pool(x)))
        x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.conv4(x))
        x = x.view(-1, 64*2*2) # Flatten layer
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(self.fc3(x),dim = 1)
        return x

def accuracy(out, labels):
    _ ,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()