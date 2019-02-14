# Semantic segmentation for volume EM

This code provides a PyTorch implementation for various semantic segmentation algorithms for volume EM: 
- CNN: Ciresan, D. C., Giusti, A., Gambardella, L. M., & Schmidhuber, J. (2012). Deep Neural Networks Segment Neuronal Membranes in Electron Microscopy Images. Neural Information Processing Systems, 1–9. https://doi.org/10.1.1.300.2221
- FCN: Shelhamer, E., Long, J., & Darrell, T. (2017). Fully Convolutional Networks for Semantic Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence. https://doi.org/10.1109/TPAMI.2016.2572683
- DeepLab: Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE Transactions on Pattern Analysis and Machine Intelligence. https://doi.org/10.1109/TPAMI.2017.2699184
- 2D U-Net: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28
- 3D U-Net: Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-net: Learning dense volumetric segmentation from sparse annotation. In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) (Vol. 9901 LNCS, pp. 424–432). https://doi.org/10.1007/978-3-319-46723-8_49

We additionally provide CRF post-processing. 

We acknowledge the code of [FCN](https://github.com/wkentaro/pytorch-fcn), [DeepLab](https://github.com/kazuto1011/deeplab-pytorch) and [CRF](https://github.com/kmkurn/pytorch-crf), which was used in this work. 

## Requirements
- Tested with Python 3.6
- Required Python libraries (these can be installed with `pip install -r requirements.txt`): 
  - torch
  - torchvision
  - numpy
  - tifffile
  - imgaug
  - scipy
  - scikit-image
  - pydensecrf (only for CRF post-processing)
  - progressbar2 (optional)
  - tensorboardX (optional)
  - jupyter (optional)
- Required data: 
  - [EPFL mitochondria dataset](https://cvlab.epfl.ch/data/data-em/)

## Usage
We provide a [notebook](train/unet.ipynb) that illustrates data loading, network training and validation. Note that the data path might be different, depending on where you downloaded the EPFL data. 
