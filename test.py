import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Viết lại model trong file train
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use1x1=False):
        super(ResidualBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)
        if use1x1:
            self.conv3=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv3 = None
    def forward(self, x):
        y=F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x=self.conv3(x)
        return F.relu(y+x)
    
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    layers = []
    if first_block:
        for i in range(num_residuals):
            layers.append(ResidualBlock(in_channels, out_channels))
    else:
        for i in range(num_residuals):
            if i == 0:
                layers.append(ResidualBlock(in_channels, out_channels, use1x1=True,stride=2))
            else:
                layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxp = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=resnet_block(64,64,2,first_block=True)
        self.layer2=resnet_block(64,128,2)
        self.layer3=resnet_block(128,256,2)
        self.layer4=resnet_block(256,512,2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

#Load mô hình, file .pth để cùng folder với file test.py 
model = torch.load("resnet18_brain_tumor.pth", weights_only=False)
model.eval()
print("Done!!")

device = torch.device("cuda")
#device = torch.device("cpu") nếu dùng cpu

def preAndShow(img_path):
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    # Chuẩn hóa ảnh (đưa giá trị về [0,1])
    image_array = image_array / 255.0
    # Chuyển sang PyTorch tensor (h,ư,c) thành (c,h,w)
    image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    # Đưa sang gpu
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)

    print(output)
    _, predicted = torch.max(output, 1)
    class_names = ["Healthy", "Tumor"]  # 0 = Healthy, 1 = Tumor
    print("Dự đoán:", class_names[predicted])
    plt.imshow(image)
    plt.title(f"Predicted: {class_names[predicted]}")
    plt.axis("off")
    plt.show()

#Ảnh nên đặt cùng folder cho đơn giản
img = "normaltest.jpeg" #Tên file ảnh ở đây
preAndShow(img)