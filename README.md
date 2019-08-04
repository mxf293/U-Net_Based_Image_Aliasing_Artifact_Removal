# Image De-aliasing
Learn and remove the 2D image aliasing artifacts caused by undersampled Fourier space using a modified U-Net structure.

### Intro
MRI raw data are measured in the Fourier space (k-space) and images are obtained by performing the inverse transform of the k-space data. Sometimes only partial k-space is sampled in order to accelerate the MRI scan. In the example shown below, the k-space is randomly undersampled by a factor of ten and it results in a ringing aliasing artifact in the image. ![](Aliased%20Image%20-%20Ground%20Truth.png)
