# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_linearmodel.py 线性模型
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.linear_model as lm

# 采集数据  读文本
x, y = np.loadtxt(
    'single.txt', delimiter=',',
    unpack=True)
# 整理输入集(二维)与输出集(一维)
x = x.reshape(-1, 1)  # 变维：n行1列
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=18)
mp.grid(linestyle=':')
mp.scatter(x, y, s=70, color='dodgerblue',
           label='Sample Points')
# 构建线性回归模型，训练模型
model = lm.LinearRegression()
# 针对训练数据，得到预测结果，画图
model.fit(x, y)
pred_y = model.predict(x)

# 评估当前模型效果
import sklearn.metrics as sm
print(sm.mean_absolute_error(y, pred_y))
print(sm.mean_squared_error(y, pred_y))
print(sm.median_absolute_error(y, pred_y))
print(sm.r2_score(y, pred_y))

mp.plot(x, pred_y, color='orangered',
        label='Regression Line')
mp.legend()
mp.show()
