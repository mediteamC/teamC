# teamC

## Team Codes and Samples:
our vm: http://35.231.196.209
login key: TeamC

## Preprocesing Pipeline:

We finalized our Pipeline as following:

--Mapping reading and merge from NACC dataset (3798 pictures)
--Normalized MRI pictures
--Get the labels and features for each pictures
  - Note that the labels in the form of array of size (L,1) where L is the number of labels
--Average pooling/cropping the MRI image
--Return all the info in the form of ndarray

Approximate time for preprocessing is around 25mins

#### https://github.com/quqixun/BrainPrep (check step 5, 6 and 7)

#### https://link.springer.com/chapter/10.1007/3-540-32390-2_64 (Paper on bias correction, to understand what it means)

## Initial Idea:

Use a classification CNN + flattening (for now arrond 4k data points) to separate AD/non AD/ MCI / Mild MCI / Other dementia using their last MRI event. Then, migrate the trained CNN to a progression task which means to classify patients that have migrated into AD or not based on their previous MRIs (arroung 1k labels) (assumption: features of the CNN should be similar for both task. Thefore, previous CNN is a good starting point + it has more data points to train)

## Final Idea:
Use a classification RESNET + flattening to separate AD/non AD/ MCI / Mild MCI / Other dementia using their last MRI event. In order to get a better trained network, we ignored the intermediate stage for the AD. That is we only focus on identifying the AD/non AD.

## Code explanation

Files: The main training code is the pre-process, models and function codes are in the folder teamc_pipline if you keep the structure it should run correctly. And the Pipline_tempete.py is the main file runs the from pre-processing to FINISH LINE.

Data: Pipline_templete.py loads NACC images from the professors directory in the supercomupter, you would need to change this if you want to load them from somewhere else. In [5]

Labels: Labels are loaded form an excel file in the main folder. Currently we are working with a 2-label balanced set. If you want to work with the original 5-label unbalanced set: In [4] change excel file name by adding "_2" at the end, In [11] modify models to 5 classes and in the data_mapper.py change the function get_label to range(5).  This was a last minute modification, so it is not coded as the rest parameters.

Training: Pipline_templete.py is the full training and printouts / stores.

Pre-process: the folder teamc_pipline contains the data mapping of labels and pictures, the cropping/downplaying and normalization and the models code.

Dictionary: in Pipline_templete.py There is a dictionary of macro-parameters. params = {'import_path':'','batch_size':30,'init_lr':0.1,'decay_freq':3,'data_length':-1,'record_name':'try','state_name':'MODEL_SAVE','epochs':1}. Import path "" means a clean new random init. training, if you put the name of a model eg. "MODEL_SAVE2" it will retake form there. I recommend to use data length 50 and batch size 20 for doing tests. This will load only the first 50 pictures (and their duplicates if you are suing the balanced set) of the training.

Model: Pipline_templete.py You can change the model you are suing but simply changing the name.  eg. resnet152() or resnet18(), I recommend using 18 for testing.

* Note that due the fact that the stampede2 doesn't officially support pytorch package, one might need to modify the code to run in parallel.

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
