# teamC

our vm: http://35.231.196.209
login key: TeamC

Link to initial paper (GANs for data augmentation): https://link.springer.com/chapter/10.1007%2F978-3-030-00320-3_4

We will replicate the classfication part of the paper for our initial task.

They are using 152 - Resnet.(I think the use cases are for classificarion)
We will also investigate voxresnet, which is also referenced in the paper. (I have only seen for image segmentation, but it should work for classification)

This is the paper for 152 resnet (explains moslty everything), its the gold standard for 3d image clasification and feature extraction: 

https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

This is the paper for voxresnet (explains moslty everything), its gold standard for 3d image segmentation: 

https://arxiv.org/pdf/1608.05895.pdf

Useful paper found on implementation in a similar task:(paper using vox CNN (not the same as voxresnet) and resnet compared for MRI and AD)
https://arxiv.org/pdf/1701.06643.pdf

Some references to read and understand the model:(RESNETS)
(Overview) https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8
(overview + recomendations) http://mynotes2ml.blogspot.com/2016/07/residual-networks-resnet.html
(overview + code) https://blog.waya.ai/deep-residual-learning-9610bb62c355

Sample codes:

resnets: (3d and 2d)

https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
https://www.kaggle.com/pytorch/resnet152/home

Voxresnets: (3d)

https://github.com/chenypic/brain_segmentation-1
https://github.com/pulimeng/3D-deep-learning
