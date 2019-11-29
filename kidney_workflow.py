import pathlib
import os
import sys
import time
import h5py
import random

import numpy as np
from scipy.spatial import distance
import keras
from keras import losses
from keras import models
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage import color, exposure, io, img_as_float, transform, filters, morphology, measure
from sklearn import metrics
from sklearn.model_selection import train_test_split
from PIL import Image
from imutils.object_detection import non_max_suppression
import cv2
from cv2 import imread, imwrite
from shutil import copyfile, make_archive, copy

from GANsegmentation.VAE import VAE
from GANsegmentation.UNet_segmentation import Unet

sys.path.append('Other segmentation methods/Wnet/src')
from WNet_bright import tf_flags, Wnet_bright
import TensorflowUtils as utils
from WNet_naive import Wnet_naive
from soft_ncut import soft_ncut, brightness_weight, gaussian_neighbor, convert_to_batchTensor
from data_io.BatchDatsetReader_VOC import create_BatchDatset, create_image_lists
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from dcrf import postprocess_dcrf

plt.rcParams['figure.figsize'] = (12.0, 9.0)
np.random.seed(42)
random.seed(123)

# target image dimensions
WIDTH = 256
HEIGHT = 256


# %%
def separate_test_data(all_image_paths, savedirname, load_prev=True):
    '''
    Separates the test data from the training data (subset of images with
    clinician-provided segmentations). In addition, remove any images belonging 
    to the same patients that are represented in the test set.
    '''
    test_savedir = savedirname + '_test'
    hold_savedir = savedirname + '_hold'
    train_savedir = savedirname + '_train'
    
    test_image_paths = []
    test_idx = []
    
    hold_image_paths = []
    hold_idx = []
    
    train_image_paths = []
    train_idx = []
    
    # get test image paths and filenames
    clinician_mask_path = 'USimages/clinician_segmented_masked_trimmed'
    mask_paths = list(pathlib.Path(clinician_mask_path).glob('*'))
    test_filenames = [os.path.splitext(os.path.basename(str(path)))[0] for path in mask_paths]
    
    # find all images that have the same subject id as any test images
    test_image_subj = [os.path.basename(str(path))[0:os.path.basename(str(path)).find('_image')] for path in mask_paths]    
    is_test_patient = [os.path.basename(str(path))[0:os.path.basename(str(path)).find('_image')] in test_image_subj for path in all_image_paths]
    
    
    if not load_prev or not os.path.isdir(test_savedir) or not os.path.isdir(hold_savedir) or not os.path.isdir(train_savedir):
        os.makedirs(test_savedir, exist_ok=True)
        os.makedirs(hold_savedir, exist_ok=True)
        os.makedirs(train_savedir, exist_ok=True)
        
        # Save test and train images in separate folders
        for i,path in enumerate(all_image_paths):
            if os.path.splitext(os.path.basename(str(path)))[0] in test_filenames:
                savepath = os.path.join(test_savedir, os.path.basename(path))
                copy(path, savepath)
                test_image_paths.append(str(path))
                test_idx.append(i)
                
            elif is_test_patient[i]:
                savepath = os.path.join(hold_savedir, os.path.basename(path))
                copy(path, savepath)
                hold_image_paths.append(str(path))
                hold_idx.append(i)
                
            else:
                savepath = os.path.join(train_savedir, os.path.basename(path))
                copy(path, savepath)
                train_image_paths.append(str(path))
                train_idx.append(i)
    else:
        # Load existing file paths and indices
        for i,path in enumerate(all_image_paths):
            if os.path.splitext(os.path.basename(str(path)))[0] in test_filenames:
                test_image_paths.append(str(path))
                test_idx.append(i)
                
            elif is_test_patient[i]:
                hold_image_paths.append(str(path))
                hold_idx.append(i)
                
            else:
                train_image_paths.append(str(path))
                train_idx.append(i)
                
    return train_image_paths, test_image_paths, hold_image_paths, np.array(train_idx), np.array(test_idx), np.array(hold_idx)


def remove_text_and_resize(image_path):
    '''
    First, we resize our cropped and despeckled ultrasound images and then apply
    the EAST text detector to remove text from any image that has annotations.
    This is done to prevent the GAN from learning to generate text on the images.
    Credit to: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
    '''
    min_confidence = 0.5 # Minimum confidence level for text detection  

    # Load and resize the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    
    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    
    # define the two output layer names for the EAST detector model
    # -- the first is the output probabilities and the second can be
    #  used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Forward pass of EAST on blobs taken from image
    blob = cv2.dnn.blobFromImage(image, 1.0, (WIDTH, HEIGHT),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Get confidence score and onstruct bounding boxes for each blob
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # apply min confidence threshold to ignore blobs
            if scoresData[x] < min_confidence:
                continue

            # Scaling correction since feature maps will be 4x smaller than input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # Compute rotation angle for prediction
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Derive dimensions of bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Computer corner points of bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Save the bounding box and correpsonding probability of text
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # Use non-max suppression to remove weak or overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Fill bounding boxes with black to cover text
    for (startX, startY, endX, endY) in boxes:
        # expand the box a little
        startX = max(0, startX - 2)
        endX = min(WIDTH - 1, endX + 2)
        startY = max(0, startY - 2)
        endY = min(HEIGHT - 1, endY + 2)
        yRange = endY - startY

        # Find the most common black colour (usually 0, but can be slightly off)
        rect = image[startX:endX, startY:endY]
        counts = np.bincount(rect.flatten())
        modeColour = int(np.argmax(counts[0:20]))

        # Fill the bounding box on the image
        cv2.rectangle(image, (startX, startY),
                      (endX, endY), (modeColour, modeColour, modeColour), cv2.FILLED)

    return image


def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()
    return None


def convert_to_png(folder, savedir):
    os.makedirs(savedir, exist_ok=True)
    
    im_paths = list(pathlib.Path(folder).glob('*.jpg'))
    for im in im_paths:
        savename = os.path.basename(im)[:-4]+'.png'
        savepath = os.path.join(savedir, savename)    
        Image.open(im).save(savepath)
    return None


def plot_edge_diagram(image_path, size=(4,4)):
    '''
    Displays and input edge diagram
    Example:
        plot_edge_diagram('USimages/coarse_diagrams_real/4_image1_us2.png')
    '''
    im = io.imread(image_path)

    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.,])
    fig.add_axes(ax)
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(im*255, cmap='gray', aspect='equal')
    return None


def plot_gen_cone(filepath, image_num):
    n = 10
    cone_size = 32
    figure = np.zeros((cone_size * n, cone_size * n))

    with h5py.File(filepath, 'r') as f:
        cones = list(f['outlines'])
        count = 0
        for i in range(n):
            for j in range(n):
                cone = (cones[image_num[count]]*255).astype(np.uint8)
                cone_outline = np.zeros_like(cone)    
                cone_outline = morphology.skeletonize(cone/255)
    
                figure[i * cone_size: (i + 1) * cone_size, 
                       j * cone_size: (j + 1) * cone_size] = cone_outline
                count += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show() 


def plot_mask_overlay(image_path, mask_paths, savename='mask_overlay.png'):
    '''
    Displays a preprocessed image (STEP 2) with output segmentation masks
    overlayed. Can accommodate up to three masks.
    '''
    clist = [(255,0,0),(0,0,255),(0,255,0)]
    im = cv2.imread(image_path, cv2.COLOR_GRAY2RGB)
    
    for i,mask_path in enumerate(mask_paths):
        # load mask
        seg = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Get mask contour
        _, contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros(seg.shape, np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        mean, _, _, _ = cv2.mean(seg, mask=mask)
        
        # Add overlay to image
#        col_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(im, contours, -1, clist[i], 2)
        overlay = cv2.fillPoly(mask, pts=contours, color=clist[i])
        im = cv2.addWeighted(overlay, 0.3, im, 1.0, 0)
        
    cv2.imwrite(savename, im)
#    cv2.imshow('img',im)
    return None


def plot_mask_overlay_ISIC(im_num='0012611'):
    '''
    Plots mask overlays for ISIC 2018 test images.
    '''
    fpath = 'Datasets/ISIC 2018/'
#    im_num = '0012611'
    image_path = fpath + '50_test_images/ISIC_' + im_num + '.jpg'
    mask_paths = [fpath + '50_predictions_using_fake_images/ISIC_' + im_num + '_segmentation.png',
                  fpath + '50_predictions_using_real_images/ISIC_' + im_num + '_segmentation.png']
    plot_mask_overlay(image_path, mask_paths, 'isic1.png')
    return None


def plot_mask_overlay_kidney(im_name='26_image1_us3'):
    '''
    Plots mask overlays for kidney ultrasound images.
    '''    
    image_path = 'STEP2_pngs/' + im_name + '.png'
    mask_paths = ['USimages/unet_segmented/' + im_name + '.png',
                  'USimages/ultra_masked_supervisedUnet/' + im_name + '.png',
                  'USimages/clinician_masks_trimmed/' + im_name + '.png',]
    plot_mask_overlay(image_path, mask_paths, 'kidney_overlay.png')
    return None
#plot_mask_overlay_kidney('830_image1_us1')


def plot_synthetic_images(fpath_synth, fpath_real):
    '''
    Plots a grid of example synthetic images drawn at random from a folder.
    Also plots a grid of real images for comparison
    '''
    ncols = 4
    nrows = 8
    
    # get random subset of images
    synth_paths = list(pathlib.Path(fpath_synth).glob('*'))
    real_paths = list(pathlib.Path(fpath_real).glob('*'))
    
    synth_images = random.sample(synth_paths, ncols*nrows)
    real_images = random.sample(real_paths, ncols*nrows)
    
    # figure generation
    im_size = 256
    figure = np.zeros((im_size*nrows, im_size*ncols*2, 3))
    fig = plt.figure(figsize=(10, 10))
    
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            rimg = io.imread(str(real_images[count]), as_gray=False)
            simg = io.imread(str(synth_images[count]), as_gray=False)

            rimg = transform.resize(rimg, (im_size,im_size,3), anti_aliasing=True)
            simg = transform.resize(simg, (im_size,im_size,3), anti_aliasing=True)
            
            figure[i*im_size:(i+1)*im_size,
                   j*im_size:(j+1)*im_size,:] = rimg
            
            figure[i*im_size:(i+1)*im_size,
                   (j+ncols)*im_size:(j+ncols+1)*im_size,:] = simg                  

            count += 1

    ax = plt.Axes(fig, [0., 0., 1., 1.,])
    fig.add_axes(ax)
    ax.set_axis_off()
    fig.add_axes(ax)
    
    plt.imshow(figure)#, cmap='Greys_r')
    plt.show()


def plot_vae_model(model):
    '''
    Plots the VAE architecture. 
    'model' can be a saved model's filepath, or a tensorflow model object.
    '''
    if isinstance(model,str):
        if os.path.isfile(model):
            model = models.load_model(model)
        else:
            print('model should be a path to a keras model')
    
    keras.utils.plot_model(model, to_file='vae_inp.png', show_shapes=True)


def load_all_images(filepath):    
    '''
    Loads the available images.
    '''
    data_root = pathlib.Path(os.path.join(filepath))
    all_image_paths = [str(path) for path in  list(data_root.glob('*'))]
    return all_image_paths


def apply_EAST(image_paths, savedir, load_prev=False):
    '''
    Apply EAST text detector and resizing for all images.
    '''
    if not load_prev or not os.path.isdir(savedir):
        os.makedirs(savedir, exist_ok=True)
        for path in image_paths:
    #        basename = os.path.basename(path)
            filename = os.path.splitext(os.path.basename(path))[0]
            savepath = os.path.join(savedir, f'{filename}.png')
            cv2.imwrite(savepath, remove_text_and_resize(path))
        
    all_image_paths = [str(path) for path in  list(pathlib.Path(savedir).glob('*'))]
    return all_image_paths


def preprocess_images(image_paths, savedir, load_prev=False):
    '''
    Rescale the intensity and boost contrast of each iamge using 
    adaptive normalization.
    '''
    if not load_prev or not os.path.isdir(savedir):
        os.makedirs(savedir, exist_ok=True)
        for path in image_paths:
            savepath = os.path.join(savedir, os.path.basename(path))
            
            im = color.rgb2gray(img_as_float(io.imread(path)))
    
            # rescale intensity and use adaptive normalization
            p2, p98 = np.percentile(im, (2, 98))
            im = exposure.rescale_intensity(im, in_range=(p2, p98))
            im = exposure.equalize_adapthist(
                np.squeeze(im), clip_limit=0.03)
        
            im = transform.resize(im, (256, 256), mode="reflect", anti_aliasing=True)
            im = color.gray2rgb(im)
            im = np.uint8(im * 255)
            io.imsave(fname=savepath,arr=im)
        
    all_image_paths = [str(path) for path in  list(pathlib.Path(savedir).glob('*'))]
    return all_image_paths


def shrink_bool_img(image, new_size):
    '''
    Shrink boolean array while retaining True values. 
    Used for downscaling edge diagrams
    '''
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
    '''
    Enlarge boolean array while retaining True values. 
    Used for upscaling edge diagrams
    '''
    im = np.empty((new_size, new_size), dtype=np.bool)
    big = im.shape[0]
    small = image.shape[0]
    scale = big // small
    for i in np.arange(small):
        for j in np.arange(small):
            im[i*scale:i*scale+scale, j*scale:j*scale +
                scale] = np.full((scale, scale), image[i, j])
    return im


def convert_to_edge_diagrams(image_paths_pre, image_paths_rcf, savedir, load_prev=True):
    '''
    Converts the RCF + NMS output into edge diagrams.
    '''
    cone_outlines = np.empty((len(image_paths_pre), 32, 32), dtype=np.bool)
    cone_outlines_path = 'cone_outlines.hdf5'
    
    if not load_prev or not os.path.isdir(savedir):
        os.makedirs(savedir, exist_ok=True)
        for (path_pre,path_rcf) in zip(image_paths_pre,image_paths_rcf):
            savepath = os.path.join(savedir, os.path.basename(path_pre))
            
            img_pre = color.rgb2gray(img_as_float(io.imread(path_pre)))
            img_rcf = color.rgb2gray(img_as_float(io.imread(path_rcf)))
        
            # Remove the outline of the cone
            binary_image = img_pre >= filters.threshold_otsu(img_pre)
            img_rcf = cv2.resize(img_rcf, (272, 272))
            img_rcf = img_rcf[8:264, 8:264]
            img_rcf *= binary_image
            img_rcf = img_rcf >= filters.threshold_otsu(img_rcf)
            
            # Shrink and simplify
            img_rcf = shrink_bool_img(img_rcf, 32)
            label_rcf = measure.label(img_rcf)
            for region in measure.regionprops(label_rcf):
                if region.area < 3:
                    for x, y in region.coords:
                        img_rcf[x, y] = False
            img_rcf = morphology.skeletonize(img_rcf)
            
            # Restore original size
            img_rcf = grow_bool_img(img_rcf, 256)
            cone = morphology.opening(img_pre, selem=morphology.disk(3))
            cone = cone >= filters.threshold_otsu(cone)
            cone = morphology.convex_hull_image(cone)
            
#            io.imsave(fname='STEP4_cone_masks', arr=(cone*255).astype(np.uint8))
            
            # Shrink cone and get its outline
            cone = shrink_bool_img(cone, 32)
            cone = (cone*255).astype(np.uint8)
            cone_outline = np.zeros_like(cone)
            cone, contours, hierarchy = cv2.findContours(cone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cone_outline, contours, -1, True, 1)
            
            cone_outline = morphology.skeletonize(cone_outline)
            cone_outlines[n] = cone_outline
            
            # Add the cone outline back to the simplified rcf output
            edge_diagram = np.logical_or(img_rcf, grow_bool_img(cone_outline, 256))
            io.imsave(fname=savepath, arr=(edge_diagram*255).astype(np.uint8))
            
            # Save the cone outlines to train the VAE later
            with h5py.File(cone_outlines_path, "w") as f:
                f.create_dataset("cone_outlines", data=cone_outlines)
                
    all_image_paths = [str(path) for path in  list(pathlib.Path(savedir).glob('*'))]
    return all_image_paths


def train_VAE_and_gen_cones(edge_image_paths, train_idx, savedir='STEP5_pngs', load_prev=True):
    '''
    Trains a VAE to generate synthetic ultrasound cones.
    '''
    train_paths = [path for i,path in enumerate(edge_image_paths) if i in train_idx]
    if not load_prev or not os.path.isdir(savedir):
        os.makedirs(savedir, exist_ok=True)
        
        # train the vae and generate cone outlnies
        vae = VAE(train_paths, savedir)
        vae.create_model()
        vae.train()
        cone_outlines = vae.generate_cones()
        
    # save generated cone outlines
    for i, cone in enumerate(cone_outlines):        
        savepath = os.path.join(savedir, '{}.png'.format(i))            
        io.imsave(fname=savepath, arr=(cone*255).astype(np.uint8))
    
    cone_image_paths = [str(path) for path in  list(pathlib.Path(savedir).glob('*'))]
    return cone_image_paths


def Generate_Masks(vae_cone_paths, savedir_mask, savedir_edge, load_prev=False):
    '''
    Generates the 'ground truth' segmentation masks for generating synthetic
    ultrasound images, and combines them with the generated cones.
    '''
    if not load_prev or not os.path.isdir(savedir_mask) or not os.path.isdir(savedir_edge):
        os.makedirs(savedir_mask, exist_ok=True)
        os.makedirs(savedir_edge, exist_ok=True)

#        n_cones = len(vae_cone_paths)
#        random.Random(42).shuffle(vae_cone_paths)
        
        # Create an edge diagram per VAE-generated cone
        for cone_path in vae_cone_paths:
            savepath_mask = os.path.join(savedir_mask, os.path.basename(cone_path))
            savepath_edge = os.path.join(savedir_edge, os.path.basename(cone_path))
            
            # Get cone outline
            outline = img_as_float(io.imread(cone_path))
            cone = morphology.convex_hull_image(outline)
        
            # Draw a big ellipse (the kidney outline) with random shape, location, and angle
            kidney_outline = np.zeros_like(outline, dtype=np.uint8)
            minor_axis = np.random.randint(5, 10)
            major_axis = np.random.randint(minor_axis + 2, minor_axis + 8)
            center_x = np.random.randint(12, 20)
            center_y = np.random.randint(8, 20)
            rotation = np.random.randint(-60, 60)
            
            # Don't draw the entire ellipse - occlusion and only partially visible outline
            angles, step = np.linspace(0, 360, np.random.randint(5, 8), retstep=True)
            for angle in angles:
                cv2.ellipse(
                    kidney_outline,
                    (center_x, center_y),
                    (major_axis, minor_axis),
                    rotation,
                    angle,
                    angle + step // 2,
                    1,
                    1,
                )
        
            # Draw a curve inside the kidney outline - kidney calyces etc.
            cv2.ellipse(
                kidney_outline,
                (center_x, center_y),
                (major_axis // 2, minor_axis // 2),
                rotation,
                30,
                120,
                1,
                1,
            )
        
            # Add some random dots within the kidney for noise - kidney internals
            kidney = np.zeros_like(kidney_outline, dtype=np.uint8)
            cv2.ellipse(kidney, (center_x, center_y),
                        (major_axis, minor_axis), rotation, 0, 360, 1, -1)
            x, y = np.where(kidney)
            i = np.random.randint(len(x), size=minor_axis * 2)
            kidney_outline[x[i], y[i]] = True
        
            kidney_outline = morphology.skeletonize(kidney_outline)
        
            # Combine kidney outline with cone outline, keeping both inside the cone
            edge_diagram = np.logical_and(np.logical_or(outline, kidney_outline), cone)
            mask = np.logical_and(kidney, cone)
            
            # resize to 256x256
            edge_diagram = grow_bool_img(edge_diagram, 256)
            mask = grow_bool_img(mask, 256)
            
            # Save synthetic masks and edge diagrams
            io.imsave(fname=savepath_mask, arr=(mask*255).astype(np.uint8))
            io.imsave(fname=savepath_edge, arr=(edge_diagram*255).astype(np.uint8))
    
    all_image_paths_mask = [str(path) for path in list(pathlib.Path(savedir_mask).glob('*'))]
    all_image_paths_edge = [str(path) for path in  list(pathlib.Path(savedir_edge).glob('*'))]
    return all_image_paths_mask, all_image_paths_edge

def compute_metrics(true_mask, pred_mask, pos_label):
    '''
    Computes the required performance metric for one test image given a 
    ground truth mask and the label in the prediction array corresponding to
    the ROI mask.
    '''
    true_mask = np.squeeze(true_mask)
    pred_mask = np.squeeze(pred_mask)
    
    if true_mask.shape != pred_mask.shape:
        true_mask = transform.resize(true_mask, pred_mask.shape, anti_aliasing=True)  
    true_mask = true_mask > 0
        
    sens = metrics.recall_score(true_mask.flatten(), pred_mask.flatten(), pos_label=pos_label)
    spec = metrics.recall_score(true_mask.flatten(), pred_mask.flatten(), pos_label=-(pos_label-1))
    acc = metrics.accuracy_score(true_mask.flatten(), pred_mask.flatten())
    f1 = metrics.f1_score(true_mask.flatten(), pred_mask.flatten(), pos_label=pos_label)
    jacc = metrics.jaccard_similarity_score(true_mask.flatten(), pred_mask.flatten(), normalize=True)
    return (sens, spec, acc, f1, jacc)


def score_model(preds, clin_path='USimages/clinician_segmented_masked_trimmed/'):
        
    scores = []
#    val_masks = [str(path) for path in list(pathlib.Path(val_path).glob('*'))]
    clin_masks = [str(path) for path in list(pathlib.Path(clin_path).glob('*'))]
    for label in np.unique(preds):
        perf = []
        for i, (ypred, path) in enumerate(zip(preds, clin_masks)):
            ytrue = img_as_float(io.imread(path))
            perf.append(compute_metrics(ytrue, ypred, label)) # (sens, spec, acc, f1, jacc)
        scores.append(perf)
    return scores


def segmentation_performance(preds, true_masks):
    '''
    Gets the performance metrics for a list of predicted segmentations and true labels.
    '''
    scores = []
#    true_masks = [str(path) for path in list(pathlib.Path(test_mask).glob('*'))]
    for label in np.unique(preds):
        perf = []
        for i, (ypred, path) in enumerate(zip(preds, true_masks)):
            ytrue = img_as_float(io.imread(path))
            perf.append(compute_metrics(ytrue, ypred, label)) # (sens, spec, acc, f1, jacc)
        scores.append(perf)
    return scores

def print_perf_stats(scores):
    '''
    Prints mean and std of performance metrics.
    '''
    for lb in range(len(scores)):
        sens = np.array([s[0] for s in scores[lb]])
        spec = np.array([s[1] for s in scores[lb]])
        accs = np.array([s[2] for s in scores[lb]])
        f1sc = np.array([s[3] for s in scores[lb]])
        jacc = np.array([s[4] for s in scores[lb]])
        print('Label {}: Sensitivity = {} +- {}'.format(lb, np.mean(sens), np.std(sens)))
        print('Label {}: Specificity = {} +- {}'.format(lb, np.mean(spec), np.std(spec)))
        print('Label {}: Accuracy    = {} +- {}'.format(lb, np.mean(accs), np.std(accs)))
        print('Label {}: F1-score    = {} +- {}'.format(lb, np.mean(f1sc), np.std(f1sc)))
        print('Label {}: Jaccard scr = {} +- {}'.format(lb, np.mean(jacc), np.std(jacc)))
    return None


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)
    return None


def use_Unet(train_path, mask_path, test_path, test_mask, savedir):
    '''
    Tests U-net for supervised image segmentation using the synthesized images.
    '''
    # Train Unet
    unet = Unet(train_path, mask_path, None)
    unet.create_model()
    unet.train()
    
    # Test on test set
    preds = unet.test(test_dir=test_path)
    
    # Compute metrics
    true_masks = [str(path) for path in list(pathlib.Path(test_mask).glob('*'))]
    scores = segmentation_performance(preds, true_masks)

    return preds, scores


def finetune_unet(train_dir, mask_dir, model):
    '''
    Takes a unet trained on synthetic images and fine-tunes it with some real
    labeled images.
    '''
    # Load the model
    unet = models.load_model('unsup_unet.h5')

    # set up the training data
    image_paths = [str(path) for path in list(pathlib.Path(train_dir).glob('*'))]
    mask_paths = [str(path) for path in list(pathlib.Path(mask_dir).glob('*'))]
    
    train_images, test_images, train_masks, test_masks =\
        train_test_split(image_paths, mask_paths, test_size=0.80, random_state=123)
    train_images, val_images, train_masks, val_masks =\
        train_test_split(train_images, train_masks, test_size=0.50, random_state=321)
#    n_train = int(np.ceil(0.1*len(image_paths)))
#    train_images, masks = zip(*random.sample(list(zip(image_paths,mask_paths)),n_train))
    
    # fine tune model
    unet.model_name = 'finetune_unet'
    unet.finetune(train_images, train_masks, val_images, val_masks)
    
    # test on unused images
    preds = unet.test(test_dir=None, test_images=test_images)
    scores = segmentation_performance(preds, test_masks)
    
    return preds, scores


def test_wnet(train_path, val_path, clin_path, savedir):
    '''
    Tests W-net for unsupervised image segmentation using the preprocessed images.
    Ues W-net with normalized cut.
    '''
    # Init Wnet model after resetting TF graph and flags
    del_all_flags(tf.flags.FLAGS)
    tf.reset_default_graph()
    wnet = Wnet_bright(tf_flags())
    
    # Prepare Data
    train_data_reader, val_data_reader = create_BatchDatset(train_path, val_path)
    
    # Train model
    wnet.train_net(train_data_reader, val_data_reader)
    
    # Test model
    test_images, preds = wnet.plot_segmentation_under_test_dir()
    
    # Postprocessing of the predictions
    post_save = [str(path) for path in list(pathlib.Path('Wnet_postpreds'))]
    [postprocess_dcrf(x1, x2, x3) for (x1,x2,x3) in zip(val_data_reader, preds, post_save)]
    # second preprocessing step, if needed, is done with the matlab toolbox: https://github.com/vrabaud/gPb
    # and overwrites the files in the Wnet_postpreds folder
    
    # Compute metrics
    postpreds = [str(path) for path in list(pathlib.Path('Wnet_postpreds').glob('*'))]
    true_masks = [str(path) for path in list(pathlib.Path(clin_path).glob('*'))]
    scores = segmentation_performance(postpreds, true_masks)
            
    return test_images, preds, scores



# %% Run Entire Pipeline

# Load all of the available images
print('Loading all images... ')
start = time.time()
all_image_paths = load_all_images('USimages/despeckled_cropped_saggital_images')
end = time.time()
print('took {} seconds'.format(end-start))

# Step 1: Apply EAST text detector and resize imaegs
print('Applying EAST and resizing all images... ')
start = time.time()
all_image_paths_STEP1 = apply_EAST(all_image_paths, savedir='STEP1_pngs', load_prev=True)
end = time.time()
print('took {} seconds'.format(end-start))

# Step 2: Preprocess images and split training and test set - gives us 918 training images, 438 test, and 1008 hold (any image with same patient ID as a test image, just in case)
print('Preprocessing all images... ')
start = time.time()
all_image_paths_STEP2 = preprocess_images(all_image_paths_STEP1, savedir='STEP2_pngs', load_prev=True)
# at this point also split train and test sets
train_image_paths_STEP2, test_image_paths_STEP2, hold_image_paths_STEP2, train_idx, test_idx, hold_idx = separate_test_data(all_image_paths_STEP2, savedirname='STEP2', load_prev=True)
end = time.time()
print('took {} seconds'.format(end-start))

# Step 3: Richer convolutional featuers (RCF) and Non-Maximum Suppression
# This step requires pretrained models in caffe and matlab, so must be done externally.
# See paper for details, also see runRCF.py and edge_nms.m
train_image_paths_STEP3 = [str(path) for path in  list(pathlib.Path('STEP3_pngs').glob('*'))]

# Step 4: Convert RCF + NMS output to edge diagrams
print('Convert RCF+NMS output to edge diagrams for all images... ')
start = time.time()
train_image_paths_STEP4 = convert_to_edge_diagrams(train_image_paths_STEP2, train_image_paths_STEP3, savedir='STEP4_pngs', load_prev=True)
end = time.time()
print('took {} seconds'.format(end-start))

# Step 5: Use a variational autoencoder to generate new cone outlines
print('Training a VAE using cones extracted from training images... ')
start = time.time()
synth_image_paths_STEP5 = train_VAE_and_gen_cones(train_image_paths_STEP4, train_idx, savedir='STEP5_pngs', load_prev=True)
end = time.time()
print('took {} seconds'.format(end-start))

# Step 6: Generate synthetic edge diagrams using VAE-generated cones
print('Creating synthetic edge diagrams with known masks... ')
start = time.time()
image_paths_STEP6_mask, image_paths_STEP6_edge = Generate_Masks(synth_image_paths_STEP5, 'STEP6_mask', 'STEP6_edge', load_prev=False)
end = time.time()
print('took {} seconds'.format(end-start))

# Step 7: Train pix2pixHD and choose the best epoch using the Frechet Inception Distance
# This uses pytorch instead of tensorflow, so we run these steps externally using the pix2pixHD code provided by nVidia
print('Training pix2pixhD with only training images... ')
start = time.time()
# Train command: python train.py --name pix2ultra --resize_or_crop none --checkpoints_dir pix2ultra/checkpoints --dataroot pix2ultra/datasets/kidneys/ --nThreads 4 --display_winsize 256 --tf_log --no_instance --label_nc 2
# FID command: python fid.py path/to/images path/to/other/images --gpu 0
end = time.time()
print('took {} seconds'.format(end-start))

# Step 8: Use the trained pix2pixHD model to generate synthetic ultrasound images from the synthetic edge diagrams
print(' all images... ')
start = time.time()
# Test command: python test.py --name pix2ultra --resize_or_crop none --checkpoints_dir pix2ultra/checkpoints --results_dir STEP8_pngs --how_many 2500 --dataroot pix2ultra/datasets/kidneys/ --display_winsize 256 --no_instance --label_nc 2
end = time.time()
print('took {} seconds'.format(end-start))

# Step 9: Use U-net to train on the synthetic images and test on real test data
print(' all images... ')
start = time.time()
train_path = 'STEP8_pngs'
mask_path = 'STEP6_mask'
test_path = 'STEP2_test'
test_mask = 'USimages/clinician_masks_trimmed'
unet_preds, unet_metrics = use_Unet(train_path, mask_path, test_path, test_mask, savedir='Unet_masks')
end = time.time()
print('took {} seconds'.format(end-start))

# Step 10: Fine-tune U-net with 43 real test images
print(' all images... ')
start = time.time()
train_dir = 'STEP2_test'
mask_dir = 'USimages/clinician_masks_trimmed'
model = 'unsup_unet.h5'
finetune_unet(train_dir, mask_dir, model)
end = time.time()
print('took {} seconds'.format(end-start))

# Step 11: Train a new U-net supervised with just real test images - 293 training, 73 val, 73 test
print(' all images... ')
start = time.time()
train_path = 'Labelled_Data_train'
mask_path = 'Labelled_Data_trainmask'
test_path = 'Labelled_Data_test'
test_mask = 'Labelled_Data_testmask'
unet_preds, unet_metrics = use_Unet(train_path, mask_path, test_path, test_mask, savedir='Unet_masks')
end = time.time()
print('took {} seconds'.format(end-start))



# OPTIONAL: Test W-net
if False: 
    print('Running test with W-net... ')
    start = time.time()
    train_path = 'STEP2_train'
    val_path = 'STEP2_test' #note that this is only used to monitor training performance, not for model selection in any way
    clin_path = 'USimages/clinician_segmented_masked_trimmed/'
    wnet_test, wnet_preds, wnet_metrics = test_wnet(train_path, val_path, clin_path, savedir='Wnet_masks')
    end = time.time()
    print('took {} seconds'.format(end-start))


