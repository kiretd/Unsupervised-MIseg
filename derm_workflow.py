import pathlib
import os
import sys
import time
import h5py
import random

import numpy as np
import tensorflow as tf
import keras
from keras import losses
from keras import models
import matplotlib.pyplot as plt

from skimage import color, exposure, io, img_as_float, transform, filters, morphology, measure
from sklearn import metrics
from PIL import Image

import cv2
from shutil import copyfile, make_archive, copy

training_input_dir = os.path.join("D:/", "Dropbox", "Datasets", "ISIC2018", "ISIC2018_Task1-2_Training_Input")
training_ground_truth_dir = os.path.join("D:/", "Dropbox", "Datasets", "ISIC2018", "ISIC2018_Task1_Training_GroundTruth")
training_rcf_nms_dir = os.path.join("D:/", "Dropbox", "Datasets", "ISIC2018", "train_rcf_nms")
validation_input_dir = os.path.join("D:/", "Dropbox", "Datasets", "ISIC2018", "ISIC2018_Task1-2_Validation_Input")
validation_rcf_nms_dir = os.path.join("D:/", "Dropbox", "Datasets", "ISIC2018", "validate_rcf_nms")
test_input_dir = os.path.join("D:/", "Dropbox", "Datasets", "ISIC2018", "ISIC2018_Task1-2_Test_Input")

# %%
def load_images(filepath):    
    '''
    Loads the available images.
    '''
    data_root = pathlib.Path(os.path.join(filepath))
    image_paths = [str(path) for path in  list(data_root.glob('*'))]
    return image_paths


def preprocess_images(image_paths, savedir, load_prev=False):
    '''
    Preprocessing for ISIC images
    '''
    if not load_prev or not os.path.isdir(savedir):
        os.makedirs(savedir, exist_ok=True)
        for path in image_paths:
            savepath = os.path.join(savedir, os.path.basename(path))
            im = img_as_float(io.imread(path))        
            im = transform.resize(im, (256, 256), mode="reflect", anti_aliasing=True)
            im = np.uint8(im * 255)
            io.imsave(fname=savepath,arr=im)
        
    all_image_paths = [str(path) for path in  list(pathlib.Path(savedir).glob('*'))]
    return all_image_paths


def shrink_bool_img(image, new_size):
    im = np.zeros((new_size, new_size), dtype=np.bool)
    big = image.shape[0]
    small = im.shape[0]
    scale = big // small
    for i in np.arange(1, small-1):
        for j in np.arange(1, small-1):
            im[i, j] = np.any(
                image[i*scale:i*scale+scale, j*scale:j*scale+scale])
    return im


def grow_bool_img(image, new_size):
    im = np.empty((new_size, new_size), dtype=np.bool)
    big = im.shape[0]
    small = image.shape[0]
    scale = big // small
    for i in np.arange(small):
        for j in np.arange(small):
            im[i*scale:i*scale+scale, j*scale:j*scale +
                scale] = np.full((scale, scale), image[i, j])
    return im


def convert_to_edge_diagrams(image_paths_rcf, savedir, load_prev=True):
    '''
    Converts the RCF + NMS output into edge diagrams.
    '''    
    if not load_prev or not os.path.isdir(savedir):
        os.makedirs(savedir, exist_ok=True)
        for path_rcf in image_paths_rcf:
            savepath = os.path.join(savedir, os.path.basename(path_rcf))
            
            img_rcf = color.rgb2gray(img_as_float(io.imread(path_rcf)))
        
            # Remove the outline of the cone
            img_rcf = cv2.resize(img_rcf, (272, 272))
            img_rcf = img_rcf[8:264, 8:264]
            img_rcf = img_rcf >= filters.threshold_otsu(img_rcf)
            
            # Shrink and simplify
            img_rcf = shrink_bool_img(img_rcf, 32)
            label_rcf = measure.label(img_rcf)
            for region in measure.regionprops(label_rcf):
                if region.area < 3:
                    for x, y in region.coords:
                        img_rcf[x, y] = False
            img_rcf = morphology.skeletonize(img_rcf)
            
            # save the output edge diagram
            edge_diagram = (img_rcf*255).astype(np.uint8)            
            io.imsave(fname=savepath, arr=edge_diagram)
                
    out_image_paths = [str(path) for path in  list(pathlib.Path(savedir).glob('*'))]
    return out_image_paths


def create_synthetic_edge_diagrams(savedir_mask, savedir_edge, n_synth=3000, load_prev=True):
    '''
    Generates the 'ground truth' segmentation masks for generating synthetic
    ultrasound images, and combines them with the generated cones.
    '''
    if not load_prev or not os.path.isdir(savedir_mask) or not os.path.isdir(savedir_edge):
        os.makedirs(savedir_mask, exist_ok=True)
        os.makedirs(savedir_edge, exist_ok=True)
        
    for n in np.arange(n_synth):
        print('Creating {} synthetic edge diagrams'.format(n_synth))
        savepath_mask = os.path.join(savedir_mask, '{}.png'.format(n))
        savepath_edge = os.path.join(savedir_edge, '{}.png'.format(n))
                     
        # Draw a big ellipse (the lesion outline) with random properties
        lesion_outline = np.zeros((32, 32))
        minor_axis = np.random.randint(2, 10)
        major_axis = np.random.randint(minor_axis + 1, minor_axis + 8)
        center_x = np.random.randint(12, 20)
        center_y = np.random.randint(12, 20)
        rotation = np.random.randint(-60, 60)
    
        # Add some random dots within the outline
        lesion = np.zeros_like(lesion_outline, dtype=np.uint8)
        cv2.ellipse(lesion, (center_x, center_y),
                    (minor_axis, major_axis), rotation, 0, 360, 1, -1)
        x, y = np.where(lesion)
        idx = np.random.randint(len(x), size=np.random.randint(minor_axis))
        lesion_outline[x[idx], y[idx]] = True
        
        # draw half of the ellipses in arcs, and half fully, just for some variation
        random_number = np.random.randint(2)
        if random_number == 0:
            # draw circle in arcs
            angles, step = np.linspace(0, 360, np.random.randint(5, 8), retstep=True)
            for angle in angles:
                cv2.ellipse(lesion_outline,
                    (center_x, center_y),
                    (minor_axis, major_axis),
                    rotation,
                    angle,
                    angle + step // 2,
                    1,
                    1,
                )
        else:
            # draw circle fully, and add some decorations
            cv2.ellipse(
                lesion_outline,
                (center_x, center_y),
                (minor_axis, major_axis),
                rotation,
                0,
                360,
                1,
                1,
            )
            # if the lesion isn't too big, maybe draw some hairs
            if (minor_axis <= 6 and major_axis <= 8):
                random_number = np.random.randint(4)
                if random_number == 0:
                    n_hairs = np.random.randint(3, 6)
                    for _ in np.arange(n_hairs):
                        start_x = np.random.randint(2, 30)
                        start_y = np.random.randint(2, 30)
                        stop_x = start_x + np.random.randint(6, 10) * (1 if np.random.random() < 0.5 else -1)
                        stop_y = start_y + np.random.randint(6, 10) * (1 if np.random.random() < 0.5 else -1)
                        cv2.line(
                            lesion_outline,
                            (start_x, start_y),
                            (stop_x, stop_y),
                            1,
                        )
            
                elif (minor_axis <= 4 and major_axis <= 6):
                    # if the lesion is small, maybe draw some other things that appear in the training set
                    random_number = np.random.randint(8)
                    if random_number == 0:
                        # draw the 2 circles on either side
                        for circle_x in [0, 32]:
                            cv2.ellipse(
                                lesion_outline,
                                (circle_x, 16),
                                (4, 10),
                                0,
                                0,
                                360,
                                1,
                                1,
                            )
                    elif random_number == 1:
                        # draw the 4 pen blots around the lesion
                        for delta_x in [8, -8]:
                            for delta_y in [8, -8]:
                                cv2.ellipse(
                                    lesion_outline,
                                    (center_x - delta_x, center_y - delta_y),
                                    (1, 1),
                                    0,
                                    0,
                                    360,
                                    1,
                                    1,
                                )
                    elif random_number == 2:
                        # draw a small ruler
                        cv2.line(
                            lesion_outline,
                            (np.random.randint(8, 12), np.random.randint(28, 30)),
                            (np.random.randint(20, 24), np.random.randint(28, 30)),
                            1,
                        )
                    elif random_number == 3:
                        # draw a big ruler
                        cv2.line(
                            lesion_outline,
                            (0, np.random.randint(26, 32)),
                            (32, np.random.randint(26, 32)),
                            1,
                        )
                    elif random_number == 4:
                        # draw the dermoscope lens
                        cv2.ellipse(
                            lesion_outline,
                            (16, 16),
                            (np.random.randint(18, 20), np.random.randint(18, 20)),
                            0,
                            0,
                            360,
                            1,
                            1,
                        )

        mask = grow_bool_img(lesion, 256)
        edge_diagram = grow_bool_img(lesion_outline, 256)
        
        mask = mask.astype(np.uint8) * 255
        edge_diagram = edge_diagram.astype(np.uint8)
        
        io.imsave(fname=savepath_mask, arr=mask)
        io.imsave(fname=savepath_edge, arr=edge_diagram)
        
    image_paths_mask = [str(path) for path in list(pathlib.Path(savedir_mask).glob('*'))]
    image_paths_edge = [str(path) for path in  list(pathlib.Path(savedir_edge).glob('*'))]
    return image_paths_mask, image_paths_edge


def use_mask_rcnn(train_path, mask_path, test_path, test_mask, savedir):
    '''
    Trains the mask_rcnn model on given images.
    '''
    # Train MRCNN
    mrcnn = MRCNN(train_path, mask_path, val_path, savedir='Models/MaskRCNN')
    mrcnn.create_model()
    mrcnn.train()
        
    return preds, scores



# %%
# Load all of the available images
print('Loading all images... ')
start = time.time()
train_image_paths = load_images(training_input_dir)
val_image_paths = load_images(validation_input_dir)
test_image_paths = load_images(test_input_dir)
end = time.time()
print('took {} seconds'.format(end-start))

# Step 1: Richer convolutional featuers (RCF) and Non-Maximum Suppression
# This step requires pretrained models in caffe and matlab, so must be done externally.
# See paper for details, as we just use two pre-packaged libraries unaltered.
# For this dataset, we don't preprocess, so we just resize the images in the RCF+NMS code
train_rcfnms_paths = [str(path) for path in list(pathlib.Path(training_rcf_nms_dir).glob('*'))]
val_rcfnms_paths = [str(path) for path in list(pathlib.Path(validation_rcf_nms_dir).glob('*'))]

# Step 2: Convert RCF + NMS output to edge diagrams
print('Convert RCF+NMS output to edge diagrams... ')
start = time.time()
train_edge_paths = convert_to_edge_diagrams(train_rcfnms_paths, savedir='train_edge', load_prev=True)
val_edge_paths = convert_to_edge_diagrams(val_rcfnms_paths, savedir='val_edge', load_prev=True)
end = time.time()
print('took {} seconds'.format(end-start))

# Step 3: Generate synthetic edge diagrams
print('Creating synthetic edge diagrams... ')
start = time.time()
synth_mask_paths, synth_edge_paths = create_synthetic_edge_diagrams('synth_mask', 'synth_edge', n_synth=3000, load_prev=True)
end = time.time()
print('took {} seconds'.format(end-start))

# Step 4: Train pix2pixHD and choose the best epoch using the Frechet Inception Distance
# This uses pytorch instead of tensorflow, so we run these steps externally using the pix2pixHD code provided by nVidia
print('Training pix2pixhD with only training images... ')
start = time.time()
# Train command: python train.py --name pix2ultra --resize_or_crop none --checkpoints_dir pix2ultra/checkpoints --dataroot pix2ultra/datasets/isic2018/ --nThreads 4 --display_winsize 256 --tf_log --no_instance --label_nc 2
# FID command: python fid.py path/to/images path/to/other/images --gpu 0
end = time.time()
print('took {} seconds'.format(end-start))

# Step 5: Use the trained pix2pixHD model to generate synthetic ultrasound images from the synthetic edge diagrams
print(' all images... ')
start = time.time()
# Test command: python test.py --name pix2ultra --resize_or_crop none --checkpoints_dir pix2ultra/checkpoints --results_dir pix2ultra/results --how_many 3000 --dataroot pix2ultra/datasets/isic2018/ --display_winsize 256 --no_instance --label_nc 2
end = time.time()
print('took {} seconds'.format(end-start))

# Step 6: Use Mask-RCNN to train on the synthetic images - test is done on competiion submission page
print(' all images... ')
start = time.time()
train_path = 'synth_isic'
mask_path = 'synth_mask'
unet_preds, unet_metrics = use_mask_rcnn(train_path, mask_path, savedir='mrcnn_masks')
end = time.time()
print('took {} seconds'.format(end-start))



