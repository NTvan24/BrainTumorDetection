import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import kagglehub
import shutil
import os
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms

# Download latest version
path = kagglehub.dataset_download("preetviradiya/brian-tumor-dataset")

print("Path to dataset files:", path)
# Thư mục đích
destination = "./brain_tumor_data"

# Tạo thư mục nếu chưa có
os.makedirs(destination, exist_ok=True)

# Di chuyển tất cả tệp từ path về destination
for file in os.listdir(path):
    shutil.move(os.path.join(path, file), os.path.join(destination, file))

print("Dataset moved to:", destination)

# Đường dẫn đến file CSV metadata 
csv_path = "./brain_tumor_data/metadata_rgb_only.csv"
# Đọc CSV 
dataset = pd.read_csv(csv_path)

#Data gồn 2 loại ảnh, ảnh chụp Não bình thường và ảnh chụp u não
tumorData=[]
normalData=[]
for (path,label) in zip(dataset["image"],dataset["class"]):
  if(label=="tumor"):
    tumorData.append([path,1])
  else:
    normalData.append([path,0])

# Đọc ảnh và resize về (224, 224)
default_tumor_path = "/content/brain_tumor_data/Brain Tumor Data Set/Brain Tumor Data Set/Brain Tumor/"
default_healthy_path = "/content/brain_tumor_data/Brain Tumor Data Set/Brain Tumor Data Set/Healthy/"

tumor_train, tumor_test = train_test_split(tumorData, test_size=0.2, random_state=42)
normal_train, normal_test = train_test_split(normalData, test_size=0.2, random_state=42)

path_train = tumor_train + normal_train
path_test = tumor_test + normal_test
X_train, y_train = [], []
X_test, y_test = [], []
for(path,label) in path_train:
    if(label==1):
      img_path=default_tumor_path + path
    else:
      img_path=default_healthy_path + path
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    # Chuẩn hóa ảnh (đưa giá trị về [0,1])
    image_array = image_array / 255.0
    X_train.append(image_array)
    y_train.append(label)
for(path,label) in path_test:
    if(label==1):
      img_path=default_tumor_path + path
    else:
      img_path=default_healthy_path + path
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    # Chuẩn hóa ảnh (đưa giá trị về [0,1])
    image_array = image_array / 255.0
    X_test.append(image_array)
    y_test.append(label)

# Chuyển thành numpy array để huấn luyện
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

#Khối phần dư
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
    
#Khối Resnet
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

#Khởi tạo mô hình Resnet18
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

# Chuyển dữ liệu thành tensor và đưa lên GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2).to('cuda') # Đổi vị trí từ batch, w, h ,channel thành batch, channel ,w ,h
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to('cuda')
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2).to('cuda')
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to('cuda')

# Tạo DataLoader
batch_size = 32
train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = data.TensorDataset(X_test_tensor, y_test_tensor)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Khởi tạo mô hình
model = ResNet18().to('cuda')

# Khai báo hàm mất mát và tối ưu
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

print("Training complete!")


#Lưu lại model
torch.save(model, "resnet18_brain_tumor.pth")

# Đánh giá mô hình
def evaluate_model(model, test_loader):
    model.eval()  # Đưa mô hình vào chế độ đánh giá
    correct = 0
    total = 0
    with torch.no_grad():  # Không cần tính toán gradient khi đánh giá
        for images, labels in test_loader:
            outputs = model(images)  # Dự đoán
            _, predicted = torch.max(outputs, 1)  # Lấy nhãn có xác suất cao nhất
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')

# Gọi hàm đánh giá
evaluate_model(model, test_loader)

#Hàm hiển thị ngẫu nhiên 1 vài kết quả dự đoán
def show_predictions(model, test_loader, num_images=5):
    model.eval() 
    images_shown = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)  
            _, predicted = torch.max(outputs, 1)
            indices = random.sample(range(len(images)), min(num_images, len(images)))  # Chọn ngẫu nhiên

            for i in indices:
                if images_shown >= num_images:
                    return
                img = images[i].cpu().numpy().transpose((1, 2, 0))
                label = labels[i].item()
                pred = predicted[i].item()
                # Hiển thị ảnh
                plt.imshow(img)
                plt.title(f"True: {'Tumor' if label == 1 else 'Normal'} | Pred: {'Tumor' if pred == 1 else 'Normal'}",
                          color="green" if label == pred else "red")  #Chữ xanh nếu đúng, đỏ nếu sai
                plt.axis("off")
                plt.show()

                images_shown += 1

show_predictions(model, test_loader, num_images=20)

