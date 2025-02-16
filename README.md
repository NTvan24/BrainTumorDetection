# BrainTumorDetection
Project phát hiện u não ( Brain Tumor Detection) trên ảnh X-Ray sử dụng mạng tích chập [Resnet18](https://www.researchgate.net/publication/364345322_Resnet18_Model_With_Sequential_Layer_For_Computing_Accuracy_On_Image_Classification_Dataset)<br>
Data được lấy từ [Kaggle](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)<br>
# Brain Tumor Detection

Dự án này sử dụng các thư viện **PyTorch**, **TensorFlow**, **scikit-learn** và **Matplotlib** để xây dựng mô hình phát hiện u não từ ảnh MRI.

## Yêu cầu hệ thống
- Python >= 3.7
- pip >= 20.0

## Cài đặt thư viện

Bạn có thể cài đặt tất cả các thư viện cần thiết bằng lệnh sau:
```bash
pip install matplotlib scikit-learn tensorflow torch numpy
```

Hoặc nếu muốn cài đặt từng thư viện riêng lẻ:
```bash
pip install matplotlib  # Vẽ biểu đồ
pip install scikit-learn  # Hỗ trợ chia dữ liệu và ML
pip install tensorflow  # Deep Learning
pip install torch  # PyTorch
pip install numpy  # Xử lý ma trận, tensor
```

## Kiểm tra cài đặt
Sau khi cài đặt xong, bạn có thể kiểm tra xem các thư viện đã được cài đặt đúng chưa bằng cách chạy lệnh:
```python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
print("Tất cả các thư viện đã được cài đặt thành công!")
```

Nếu không có lỗi nào xuất hiện, bạn đã sẵn sàng để chạy dự án!

---

🚀 **Tiếp theo:** Chạy mô hình bằng lệnh:
```bash
python train.py
```


 
