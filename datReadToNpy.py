# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:10:07 2024

@author: Shangyang
"""

import numpy as np
import h5py  # 需要安装h5py库
from scipy.io import savemat

# 设置文件名
filename = 'continuous.dat'

# 数据参数
n_channels = 384  # 通道数，具体数值根据您的设备和配置决定
dtype = np.int16  # 数据类型，根据实际情况选择

# 计算每个数据样本的大小（以字节为单位）
sample_size = np.dtype(dtype).itemsize * n_channels

# 读取数据
data = np.memmap(filename, dtype=dtype, mode='r')

# 将数据重塑为 [时间, 通道] 格式，然后转置为 [通道, 时间]
lfp_data = data.reshape((-1, n_channels)).T

# 输出一些数据进行检查
print(lfp_data.shape)
print(lfp_data[:10])  # 打印前10个时间点的数据

np.save('lfp_data.npy',data)


# 在 MATLAB 中读取 HDF5 文件：
# 在 MATLAB 中，你可以直接使用 load 函数来读取这个文件，就像读取传统的 .mat 文件一样：# 
# data = load('lfp_data.mat');
# disp(data.lfp_data);


