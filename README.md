# 实验二
## 1
### 
四大数据增广的方法分别做了以下操作：

  水平翻转：将图像水平翻转，实现左右镜像。

  垂直翻转：将图像垂直翻转，实现上下镜像。

  随机旋转：将图像随机旋转一定角度，从而增加图像的角度差异性。
  
  随机裁切：随机裁剪图像的一部分，从而增加图像的位置差异性。
   
  随机色度变换：对图像的色度进行一定的随机变换，从而增加图像的颜色差异性。

它们分别要解决以下视觉问题：

  水平翻转和垂直翻转可以解决平移不变性问题。
 
  随机旋转可以解决旋转不变性问题。
  
  随机裁切可以解决尺寸不变性问题。
   
  随机色度变换可以解决光照复杂性问题。
  
  ## 2
  ### 实现以下数据增广:
1.随机水平翻转，概率50%

2.随机旋转，角度范围要小，概率50%

3.随机缩放裁切，裁切后尺寸256，概率50%

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# 定义数据增广函数
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=100,expand=False),
    transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
    transforms.ToTensor(),
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
