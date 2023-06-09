# 实验二
## 1、原理浅究
### 四大数据增广的方法分别做了以下操作：

   #### 水平翻转：将图像水平翻转，实现左右镜像。  
   

   #### 垂直翻转：将图像垂直翻转，实现上下镜像。  
   

   #### 随机旋转：将图像随机旋转一定角度，从而增加图像的角度差异性。  
   
    
   #### 随机裁切：随机裁剪图像的一部分，从而增加图像的位置差异性。  
   
   
   #### 随机色度变换：对图像的色度进行一定的随机变换，从而增加图像的颜色差异性。  
   

### 它们分别要解决以下视觉问题：

   #### 水平翻转和垂直翻转可以解决平移不变性问题。  
   
  
   #### 随机旋转可以解决旋转不变性问题。  
   
  
   #### 随机裁切可以解决尺寸不变性问题。  
   
   
   #### 随机色度变换可以解决光照复杂性问题。  
   
   
  
  ## 2、数据增广
  
  ### 好处：
  
  #### 通过数据增广，可以扩充训练数据集，增加数据集的多样性，提高模型的[泛化能力]。此外，数据增广还可以防止过拟合，提高模型的[鲁棒性]。在实际应用中，数据增广是非常重要的技术，可以帮助我们训练出更加准确和鲁棒的模型。
  
  ### 实现以下数据增广:
#### 1.随机水平翻转，概率50%

#### 2.随机旋转，角度范围要小，概率50%

#### 3.随机缩放裁切，裁切后尺寸256，概率50%
### 1、对单一图片剪切
```python
import torch
import torchvision.transforms as transforms
from PIL import Image      #使用Pillow库中的Image模块，可以打开、创建、保存、缩放、裁剪、旋转和转换图像格式等操作

# 定义数据增广函数
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),     #以0.5的概率随机水平左右翻转图像
    transforms.RandomRotation(degrees=100,expand=False),      #以给定的角度随机旋转输入的图像
    transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0), ratio=(0.8, 1.2)),    #用于随机裁剪和缩放图像
    transforms.ToTensor(),         #用于将PIL图像或NumPy数组转换为尺寸为(C, H, W)的张量
])

# 加载图片
image = Image.open('dog.jpg')

# 对图片进行增广
augmented_image = transform(image)

# 显示增广后的图片
import matplotlib.pyplot as plt
plt.imshow(augmented_image.permute(1, 2, 0))
plt.show()
```
### 训练集
增广前

![dog](https://user-images.githubusercontent.com/119326710/228847735-749da688-78b2-48d4-934c-2d7fe419870e.jpg)

增广后

![dog1](https://user-images.githubusercontent.com/119326710/228848274-cf89147b-5bfd-41cd-8447-30717dbb82ca.jpg)


### 2、对训练＋验证照片剪切
```python
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义数据增强变换
transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.5, 0.5), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor(),
])

# 加载训练集图片和验证集图片
train_image = Image.open(r'D:\data  augmentation\classification-basic-sample-master\新建文件夹\train\dog.jpg')
val_image = Image.open(r'D:\data  augmentation\classification-basic-sample-master\新建文件夹\val\lb.jpg')

# 对训练集图片和验证集图片进行增广
augmented_train_image = transform(train_image)
augmented_val_image = transform(val_image)

# 显示增广后的图片
fig, axs = plt.subplots(1, 2)
axs[0].imshow(augmented_train_image.permute(1, 2, 0))
axs[1].imshow(augmented_val_image.permute(1, 2, 0))
plt.show()
```
### 训练集

增广后

![test2](https://user-images.githubusercontent.com/119326710/228850465-d38327d1-f5ac-41b7-a7c0-903cd52bde8d.jpg)


