# **智能眼底影像分析**

[原始数据集](https://drive.google.com/drive/folders/1H1xD5riEKEmm_riY_nZoa4Yesi6OS7Rv?usp=sharing)、[模型权重](https://drive.google.com/drive/folders/1cdNCKqMRbC4SzH3-bI2en9IaWtvmGPO2?usp=sharing)

## 任务一：中心凹检测定位

### 模型简述

​        在此任务中我们选择了以YOLOv5模型为基础进行眼底图像中心凹的定位检测。YOLO模型发展到最新的第五代已经算是一个非常成熟的体系了，相关的数据预处理、预训练权重、调参流程和结果可视化都做得比较完备了，这也是我们选择使用它进行目标检测任务的一个原因。

​		另外一点原因则是由于项目本身的特性。我们观察原始数据集后不难看出，每幅图像只有一个检测目标，并且标定框的尺寸相对于整幅图片来说也不是很小，因此即使是使用单阶段检测方法也能获得不错的效果，此外我们也考虑过使用基于注意力机制的端到端检测方法，但是由于标定数据集较少我们最终还是决定使用YOLOv5进行训练。

![](E:\git repositories\Fundus-Imaging-Diagnosis\20210429093338824.png)

​		YOLOv5在三代和四代的基础上做出了一些调整。在输入端除了保留Mosaic数据增强方法、它还增加了自适应锚框计算、自适应图片缩放等功能。backbone方面，v5在输入端后的降采样方式更新为Focus模块并且调整了CSP的内部结构。在neck层级中保留了v4的融合多尺度特征图的方式。

### 训练流程

[数据集](https://drive.google.com/drive/folders/1T54Cn1Y98KO_SauicJ22tQQ20ZGFlTK3?usp=sharing)



### 实验和结果分析



## 任务二：血管分割

### 模型简述



### 训练流程

[数据集](https://drive.google.com/drive/folders/16Usia2gUBUzLglrNI2edM5iJaclvrZrj?usp=sharing)

### 实验和结果分析



## 任务三：糖尿病视网膜病变分级

### 模型简述



### 训练流程

[数据集](https://drive.google.com/drive/folders/1T54Cn1Y98KO_SauicJ22tQQ20ZGFlTK3?usp=sharing)

### 实验和结果分析

------

参考项目：

- https://github.com/ultralytics/yolov5.git

- https://github.com/WZMIAOMIAO/deep-learning-for-image-processing.git
