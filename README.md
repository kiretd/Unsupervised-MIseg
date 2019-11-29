# Unsupervised-MIseg 
This repository contains the code to implement unsupervised medical image segmentation using edge mapping and adversarial learning as described in our paper: [Unsupervised Medical Image Segmentation with Adversarial Networks: From Edge Diagrams to Segmentation Maps](https://arxiv.org/abs/1911.05140).

Note that this version was written so that it is easier to follow the logic of the approach. It is therefore meant to be used for scripting. It should be fairly easy to adapt it for different purposes, but we may add other versions for different kinds of users (especially if requested). 

# Requirements
For now, different parts of this project were adapted from different libraries, and so the dependencies vary by which step of the process is being run. The following are needed to run all parts of the code:

- Python 3
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Caffe](https://caffe.berkeleyvision.org/)
- [Pytorch](https://pytorch.org/)
- [Matlab](https://www.mathworks.com/products/matlab.html)
- [scipy](https://www.scipy.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Scikit-image](https://scikit-image.org/)
- [Open-CV](https://github.com/skvark/opencv-python)

# Additional Resources (borrowed and adapted code)
We use models and algorithms from the following repositories:

- [nVidia's pix2pixHD](https://github.com/NVIDIA/pix2pixHD) for our GAN architecture
- [Richer Convolutional Features](https://github.com/yun-liu/rcf) for edge detection
- [Structured Edge Detection](https://github.com/pdollar/edges) for skeletonizing the output of RCF
- [Mask R-CNN](https://github.com/matterport/Mask_RCNN) for the ISIC 2018 segmentation model

Comparison with W-net:

- [This W-net repo](https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut) for our W-net comparison
- We adapted the Dense CRF from [this repo](https://github.com/lucasb-eyer/pydensecrf) for our dcrf.py implementation, used for post-processing the W-net outputs.
- We also used [this](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) for segmentation-grouping to clean up the output (and consolidate segmentation classes for the W-net)

# Using this method
Most of the functions can be kept as is and easily adapted to other datasets. However ,the following two things need to be done for each dataset:

- Preprocess the images according to your needs. Our models currently take images in 256x256 pixels, but that can be changed in the code and the models can be retrained.
- You will need to write a custom function for construction edge diagrams according to the shape of the objects that need to be segmented in your images.
- Once these two steps are done, follow the steps laid our in one of the workflow.py files to train the GAN and your preferred segmentation model.

# Some results from our paper
### Real (first four columns) vs. Synthetic (last four columns) kidney ultrasounds

![synthetic kidney ultrasounds][synth_kidney]

### Real (first four columns) vs. Synthetic (last four columns) ISIC skin lesion images

![synthetic ISIC_images][synth_skin]

### Real test kidney ultrasound segmentation results (green: clinician-provided labels, red: supervised model, blue: our unsupervised approach)

![kidney seg][kidney_results]

### Real test ISIC skin lesion segmentation results (red: supervised model, blue: our unsupervised approach)

![ISIC_seg][isic_results]

[synth_kidney]: https://github.com/kiretd/Unsupervised-MIseg/blob/master/sample_images/synth_kidneys.png
[synth_skin]: https://github.com/kiretd/Unsupervised-MIseg/blob/master/sample_images/synth_skin.png
[kidney_results]: https://github.com/kiretd/Unsupervised-MIseg/blob/master/sample_images/kidney_results_figure.png
[isic_results]: https://github.com/kiretd/Unsupervised-MIseg/blob/master/sample_images/isic_results_figure.png

