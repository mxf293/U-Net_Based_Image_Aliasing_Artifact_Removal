# Image De-aliasing
Learn and remove the 2D image aliasing artifacts caused by undersampled Fourier space using a modified U-Net structure.

### Introduction
MRI raw data are measured in the Fourier space (k-space) and images are obtained by performing the inverse transform of the k-space data. Sometimes only partial k-space is sampled in order to accelerate the MRI scan. In the example shown below, the k-space is randomly undersampled by a factor of ten and it results in a ringing aliasing artifact in the image. The objective of this work is to remove the aliasing artifact in the image and recover the clear image using a deep learning model. ![](Aliased%20Image%20-%20Ground%20Truth.png)

### Data Source
The data used in this program is download from the online [IXI Dataset](http://brain-development.org/ixi-dataset/). 20,000 T2-weighted images from 200 subjects (100 different slices per subject) are gathered for the following demonstration. 

### Deep Learning Structure: Modified U-Net
The model is
