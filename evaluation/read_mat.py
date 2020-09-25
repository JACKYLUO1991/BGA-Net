#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 13:23
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : read_mat.py
# @Software: PyCharm

import pandas as pd
import scipy.io as scio

dataFile = "roc_curves/CUHKMED/roc_curve.mat"
data = scio.loadmat(dataFile)

fpr = data['fpr'][0]
tpr = data['tpr'][0]

df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
df.to_csv("res.csv")
