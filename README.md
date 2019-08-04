# Image De-aliasing
Learn and remove the 2D image aliasing artifacts caused by undersampled Fourier space using a modified U-Net structure.

### Introduction
MRI raw data are measured in the Fourier space (k-space) and images are obtained by performing the inverse transform of the k-space data. But in order to accelerate the MRI scan, some recent studies start to sample only a small portion of the k-space. In the example shown below, the k-space is randomly undersampled by a factor of ten and it results in a ringing aliasing artifact in the image. The objective of this work is to remove the aliasing artifact in the image and recover the clear image using a deep learning model. 
![](Aliased%20Image%20-%20Ground%20Truth.png)

### Data Source
The data used in this program is download from the online [IXI Dataset](http://brain-development.org/ixi-dataset/). 20,000 T2-weighted images from 200 subjects (100 different slices per subject) are gathered for the following demonstration. 

### Deep Learning Structure: Modified U-Net
The model employed in this work is based on [U-Net](https://arxiv.org/abs/1505.04597). The specific structure consists a down-convolution path and an up-convolution path with skip connections. The idea is that the features extracted during the down-sampling path as well as the up-convolution path are all used to recover an aliasing free image.
<img src="https://github.com/mxf293/Image_De-aliasing/blob/master/Model%20Structure.jpg" width="1000" height="100">

### Results
