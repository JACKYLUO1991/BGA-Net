# BGA-Net
Joint optic disc and optic cup segmentation based on boundary prior and adversarial learning.

## Purpose 
The most direct means of glaucoma screening is to use cup-to-disc ratio (CDR) via color fundus photography, the first step of which is the precise segmentation of the optic cup (OC) and optic disc (OD). In recent years, convolution neural networks (CNN) have shown outstanding performance in medical segmentation tasks. However, most CNN-based methods ignore the effect of boundary ambiguity on performance, which leads to low generalization. This paper is dedicated to solving this issue.

## Methods 
In this paper, we propose a novel segmentation architecture, called BGA-Net, which introduces an auxiliary boundary branch and adversarial learning to jointly segment OD and OC in a multi-label manner. To generate more accurate results, the generative adversarial network (GAN) is exploited to encourage boundary and mask predictions to be similar to the ground truth ones.

## Results 
Experimental results show that our BGA-Net system achieves stateof-the-art OC and OD segmentation performance on three publicly available datasets, i.e., the Dice scores for the optic disc/cup on the Drishti-GS, RIMONE-r3 and REFUGE datasets are 0.975/0.898, 0.967/0.872 and 0.951/0.866, respectively.

## Conclusion
In this work, we not only achieve superior OD and OC segmentation results, but also confirm that the values calculated through the geometric relationship between the former two are highly related to glaucoma.

#### For training：sh train_Drishti-GS.sh

#### Please cite:
```
@article{luo2021joint,
  title={Joint optic disc and optic cup segmentation based on boundary prior and adversarial learning},
  author={Luo, Ling and Xue, Dingyu and Pan, Feng and Feng, Xinglong},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--10},
  year={2021},
  publisher={Springer}
}
```
