# teamC

## Team Codes and Samples:
our vm: http://35.231.196.209
login key: TeamC

## Preprocesing Pipeline:

This should be --> mapping reading and merge (NACC + OASIS) (ok) - sampling for mini batch (ok) - create label one hot encode Y tensor (ok) - MRI registration (Read point 5) - MRI Skull Stripping (Read Point 6) - MRI Bias Field Correction (read point 7) - MRI Cropping (Waiting for vaib max cropping) - MRI range standarization (OK) - Add MRI cube to X tensor (OK) - Finish when all the minibatch MRI processed images are in the X tensor (OK).

To finish the cleaning pipeline, we should define the max cropping, and also we should look into skull stripping because it is used in every mayor paper i have checked. see some code / example below. Vaib, you will probably need to run the max crop again because it should be done after some preprocesing.

#### https://github.com/quqixun/BrainPrep (check step 5, 6 and 7)

#### https://link.springer.com/chapter/10.1007/3-540-32390-2_64 (Paper on bias correction, to understand what it means)

## Initial Idea:

Use a classification CNN + flattening (for now arrond 6k data points) to separate AD/non AD/ MCI / Mild MCI / Other dementia using their last MRI event. Then, migrate the trained CNN to a progression task which means to classify patients that have migrated into AD or not based on their previous MRIs (arroung 1k labels) (assumption: features of the CNN should be similar for both task. Thefore, previous CNN is a good starting point + it has more data points to train)

## Classification Task:

#### Link to initial paper (GANs for data augmentation): https://link.springer.com/chapter/10.1007%2F978-3-030-00320-3_4

We will replicate the classfication part of the paper for our initial task.

They are using 152 - Resnet.(I think the use cases are for classificarion)
We will also investigate voxresnet, which is also referenced in the paper. (I have only seen for image segmentation, but it should work for classification)

This is the paper for 152 resnet (explains moslty everything), its the gold standard for 3d image clasification and feature extraction: 

#### https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

This is the paper for voxresnet (explains moslty everything), its gold standard for 3d image segmentation: 

#### https://arxiv.org/pdf/1608.05895.pdf

Useful paper found on implementation in a similar task:(paper using vox CNN (not the same as voxresnet) and resnet compared for MRI and AD)
#### https://arxiv.org/pdf/1701.06643.pdf

Some references to read and understand the model:(RESNETS)
#### (Overview) https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8
#### (overview + recomendations) http://mynotes2ml.blogspot.com/2016/07/residual-networks-resnet.html
#### (overview + code) https://blog.waya.ai/deep-residual-learning-9610bb62c355

## Sample codes:

resnets: (3d and 2d)

#### https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
#### https://www.kaggle.com/pytorch/resnet152/home

Voxresnets: (3d)

#### https://github.com/chenypic/brain_segmentation-1
#### https://github.com/pulimeng/3D-deep-learning
