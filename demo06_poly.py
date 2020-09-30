# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_poly.py 多项式回归
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.linear_model as lm
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
# 采集数据  读文本
x, y = np.loadtxt(
    'single.txt', delimiter=',',
    unpack=True)
# 整理输入集(二维)与输出集(一维)
x = x.reshape(-1, 1)  # 变维：n行1列
mp.figure('Poly Regression', facecolor='lightgray')
mp.title('Poly Regression', fontsize=18)
mp.grid(linestyle=':')
mp.scatter(x, y, s=70, color='dodgerblue',
           label='Sample Points')

# 多项式回归
model = pl.make_pipeline(
    sp.PolynomialFeatures(10), lm.LinearRegression())
model.fit(x, y)
pred_y = model.predict(x)

# 评估当前模型效果
import sklearn.metrics as sm
print(sm.mean_absolute_error(y, pred_y))
print(sm.mean_squared_error(y, pred_y))
print(sm.median_absolute_error(y, pred_y))
print(sm.r2_score(y, pred_y))
# 绘制多项式函数图像，从min到max拆500个点，
# 预测500个函数值，按顺序连线。
x = np.linspace(x.min(), x.max(), 500)
pred_y = model.predict(x.reshape(-1, 1))
mp.plot(x, pred_y, color='orangered',
        label='Regression Line')
mp.legend()
mp.show()
