Hi this is the file explains the process for teamc_pipline package
Basically it contains the following file


data_mapper # The parser for data preprocessing
resnetmodel # Resnet model library
Resnet3D # The 3d version
trainning_recorder # Helper function which is working on performance analysis

The data_mapper is the main module you will be using in Pipline_templete.py file
It first takes an excel file that mappes the label and MRI file names.
Then take the exact file from tar folder and generate a huge ND array =》 data shape is in (N,Channel,size,size,size) labels shape is in (N,-1)

