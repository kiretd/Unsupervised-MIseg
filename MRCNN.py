import os
import pathlib
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from imgaug import augmenters as iaa
from PIL import Image

from sklearn.model_selection import train_test_split

sys.path.append('ISIC2018/Mask_RCNN')
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


class ShapesConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides values specific
    to the dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each
    # GPU. Batch size is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + lesion

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # set number of epoch
    STEPS_PER_EPOCH = 500

    # set validation steps 
    VALIDATION_STEPS = 50
    

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    

class ISIC_Dataset(utils.Dataset):
    
    def load_shapes(self, mode, train_paths=None, val_paths=None):
        
        # Add classes
        self.add_class("shapes", 1, "lesion")
        
        if mode == "train":  
            for im in train_paths:
                self.add_image('shapes', image_id=os.path.basename(im), path=im)             
              
        if mode == "val":
            for im in val_paths:
                self.add_image('shapes', image_id=os.path.basename(im), path=im)

    def image_reference(self, image_id):
        '''
        Return the shapes data of the image.
        '''
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id, mask_dir):
        '''
        Generate instance masks for shapes of the given image ID.
        '''
        info = self.image_info[image_id]        
        path = os.path.join(mask_dir, os.path.splitext(info.get("id"))[0] + "_segmentation.png")
        mask = imread(path, pilmode='L')
        mask = np.expand_dims(mask, axis=-1)
        
        # Map class names to class IDs.
        class_ids = np.ones((1,), dtype=np.int32)
        
        return mask, class_ids


class MRCNN():
    def __init__(self, train_dir, mask_dir, savedir='Models/MaskRCNN', 
                 model_name='mrcnn_new',
                 init_weights='coco',
                 im_size=(256,256,3),
                 batch_size=8,
                 epochs=100):
        '''
        Initialize the MRCNN object parameters.
        '''
        self.train_dir = train_dir
        self.mask_dir = mask_dir
#        self.val_dir = val_dir
        self.savedir = savedir
        self.im_size = im_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name + '.h5'
        self.init_weights = init_weights
        
        if not os.path.exists(savedir):
            os.makedirs(savedir)

    def setup_data_paths(self):
        '''
        Obtains image and mask paths from directories.
        '''
        train_dir = self.train_dir
        mask_dir = self.mask_dir
#        val_dir = self.val_dir
        
        train_images = [str(path) for path in list(pathlib.Path(train_dir).glob('*.png'))]
        train_masks = [str(path) for path in list(pathlib.Path(mask_dir).glob('*.png'))]
#        val_images = [str(path) for path in list(pathlib.Path(val_dir).glob('*.png'))]
        
        train_images, train_masks, val_images, val_masks =\
            train_test_split(train_images, train_masks, test_size=0.2, random_state=42)
        
        print('Number of training examples: {}'.format(len(train_images)))
        print('Number of validation examples: {}'.format(len(val_images)))
        
        self.train_images = train_images
        self.train_masks = train_masks
        self.val_images = val_images
        self.val_masks = val_masks
        self.n_train = len(train_images)
        self.n_val = len(val_images)
        
    def _process_pathnames(self, fname, label_path):
        '''
        Load input image.
        '''
        img_str = tf.read_file(fname)
        img = tf.image.decode_png(img_str, channels=3)
        
        label_img_str = tf.read_file(label_path)
        label_img = tf.image.decode_png(label_img_str, channels=1)
        return img, label_img
    
    def load_model(self):
        '''
        Loads pretrained COCO weights.
        '''
        savedir = self.savedir
        init = self.init_weights
        
        # Create model in training mode
        config = ShapesConfig()
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=savedir)
            
        # Choose weights for init
        if init == 'coco':
            coco_path = os.path.join(savedir, 'mask_rcnn_coco.h5')    
            if not os.path.exists(coco_path):
                utils.download_trained_weights(coco_path)
                   
            model.load_weights(coco_path, by_name=True,
                               exclude=['mrcnn_class_logits', 
                                        'mrcnn_bbox_fc',
                                        'mrcnn_bbox', 
                                        'mrcnn_mask'])
    
        elif init == 'imagenet':
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        
        elif init == 'last':
            model.load_weights(model.find_last(), by_name=True)
        
        self.model = model   
        self.config = config
        return model
    
    def prepare_dataset(self):
        '''
        Prepares a dataset for training or validation.
        '''
        self.setup_datapaths()
        
        train_images = self.train_images
        val_images = self.val_images
        
        train = ISIC_Dataset()
        train.load_shapes('train', train_paths=train_images)
        train.prepare()
        
        val = ISIC_Dataset()
        val.load_shapes('val', train_paths=None, val_paths=val_images)
        val.prepare()
        
        return train, val
        
    def train(self):
        '''
        Train the loaded model
        '''
        config = self.config
        epochs = self.epochs
        
        # get data and model
        traindata, valdata = self.prepare_dataset()
        model = self.load_model()
        
        # hardcoded augmentation approach for now
        augmentation = iaa.SomeOf((0, 2), [
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.OneOf([iaa.Affine(rotate=90),
                               iaa.Affine(rotate=180),
                               iaa.Affine(rotate=270)]),
                    iaa.Multiply((0.8, 1.5)),
                    iaa.GaussianBlur(sigma=(0.0, 5.0))
                    ])
        
        # train model
        model.train(traindata, valdata,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs,
                    augmentation=augmentation,
                    layers='all')
        
        self.model = model
        self.augmentation = augmentation
        
    def inference(self):
        '''
        Runs model in inference mode.
        '''
        config = InferenceConfig()
        model = self.model
        savedir = self.savedir
        
        model = modellib.MaskRCNN(mode='inference',
                                  config=config,
                                  model_dir=savedir)
        
        self.model = model
        self.inference_config = config

