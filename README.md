# Image De-aliasing
Learn and remove the 2D image aliasing artifacts caused by undersampled Fourier space using a modified U-Net structure.

### Introduction
MRI raw data are measured in the Fourier space (k-space) and images are obtained by performing the inverse transform of the k-space data. But in order to accelerate the MRI scan, some recent studies start to sample only a small portion of the k-space. In the example shown below, the k-space is randomly undersampled by a factor of ten and it results in a ringing aliasing artifact in the image. The objective of this work is to remove the aliasing artifact in the image and recover the clear image using a deep learning model.

<img src="https://github.com/mxf293/Image_De-aliasing/blob/master/Aliased%20Image%20-%20Ground%20Truth.png" width="600" height="300">

### Model Input and Output Data
The data used in this program are from the online [IXI Dataset](http://brain-development.org/ixi-dataset/). 20,000 T2-weighted images from 200 subjects (100 different slices per subject) are gathered for the following demonstration. Aliased images are generated by applying a sparse k-space mask to the Fourier transform of the ground truth image and then performing the inverse Fourier transform.

<img src="https://github.com/mxf293/Image_De-aliasing/blob/master/Aliased%20Data%20Synthesis.png" width="600" height="400">

### Deep Learning Structure: Modified U-Net
The model employed in this work is based on [U-Net](https://arxiv.org/abs/1505.04597). The specific structure consists a down-convolution path and an up-convolution path with skip connections. The idea is that the features extracted during the down-sampling path as well as the up-convolution path are all used to recover an aliasing free image. The convolutional layers are 3 by 3 filters with ReLU activation function and zero-paddings. The upsampling factor is 2 with bi-linear interpolations.The original U-Net is developed for object segmentation where it is a pixel-wise binary classification problem and therefore softmax activation function is used in the last layer. In this work, the activation function in the last layer is linear so that it outputs continuous values. The loss function is the mean square error. Adam optimizer is chosen with a learning rate of 1e-3.

<img src="https://github.com/mxf293/Image_De-aliasing/blob/master/Model%20Structure.jpg" width="520" height="400">

### Training
The dataset is split into 95% training set and 5% test set. In order to monitor and prevent overfitting, the training set is further divided into 95% actual training set and 5% validation set. Since no regularizer is incorporated, early stopping is used.

### Results
