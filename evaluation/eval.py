#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/12 18:17
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : eval.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

# files = ["Drishti-GS-results.csv", "RIM-ONE_r3-results.csv"]
files = ['refuge-test-results.csv']
FPRs = []
TPRs = []
AUCs = []

for i , file in enumerate(files):
    df_cdr = pd.read_csv(file, usecols=['CDR'])
    df_glau = pd.read_csv(file, usecols=['Glaucoma'])

    df_cdr = df_cdr.values.tolist()
    df_glau = df_glau.values.tolist()

    fpr, tpr, _ = roc_curve(df_glau, df_cdr)
    roc_auc = auc(fpr, tpr)

    FPRs.append(fpr)
    TPRs.append(tpr)
    AUCs.append(roc_auc)

    # df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    # df.to_csv(str(i) + "res.csv")

for fpr, tpr, roc_auc in zip(FPRs, TPRs, AUCs):
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.xlim([0.0, 0.5])
plt.ylim([0.5, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
