README:

Super CRF toolbox for training an conditional random field in combination with super pixel annotations
=================


Folders:
=================

SuperCRF: 
============
Matlab scripts for the training of a SuperCRF, that combines Superpixel annotations in different scales as well as single class classifications.


The CRF is based in the UGM toolbox (https://www.cs.ubc.ca/~schmidtm/Software/UGM.html)

RegionClassification: 
=========================
Matlab functions to extract features from each superpixel and use them to train a SVM model to classify them into regions.

To train the SVM the 'libsvm' library (version 3.22) is needed:

https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download
https://github.com/cjlin1/libsvm

The following Matlab packages will be needed for feature extraction:

1) Haralick features:
https://uk.mathworks.com/matlabcentral/fileexchange/58769-haralicktexturefeatures

2) SFTA (Segmentation-based Fractal Texture Analysis) features:
https://uk.mathworks.com/matlabcentral/fileexchange/37933-alceufc-sfta


Dataset:
============
Sample barcodes from TCGA that were used for training and testing of the SuperCRF.
The images can easily be accessed using the GDC client and TCGA (https://portal.gdc.cancer.gov/).
