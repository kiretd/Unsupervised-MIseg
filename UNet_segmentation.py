import pathlib
import os
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

from skimage import io

import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
#from tensorflow.python.keras import backend as K  

def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder
    
class Unet():
    def __init__(self, train_dir, mask_dir, savedir='Unet', model_name='unet_new',
                 im_size=(256,256,3),
                 batch_size=8,
                 epochs=100,
                 val_prop=0.20):
        '''
        Initialize the Unet object parameters.
        '''
        self.train_dir = train_dir
        self.mask_dir = mask_dir
        self.savedir = savedir
        self.im_size = im_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_prop = val_prop
        self.model_name = model_name + '.h5'
        
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        
    def setup_data_paths(self):
        '''
        Obtains image and mask paths from directories.
        '''
        train_dir = self.train_dir
        mask_dir = self.mask_dir
        val_prop = self.val_prop
        
        images = [str(path) for path in list(pathlib.Path(train_dir).glob('*.png'))]
        masks = [str(path) for path in list(pathlib.Path(mask_dir).glob('*.png'))]
        
        N = len(images)
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        train_indices = indices[:int(np.floor(N*val_prop))]
        val_indices = indices[int(np.floor(N*val_prop)):]
        
        train_images = [im for i, im in enumerate(images) if i in train_indices]
        train_masks = [msk for i, msk in enumerate(masks) if i in train_indices]
        
        val_images = [im for i, im in enumerate(images) if i in val_indices]
        val_masks = [msk for i, msk in enumerate(masks) if i in val_indices]
        
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
    
    def shift_img(self, output_img, label_img, width_shift_range, height_shift_range):
      '''
      Perform horizontal or vertical shifts for data augmentation.
      '''
      im_size = self.im_size
      if width_shift_range or height_shift_range:
          if width_shift_range:
              width_shift_range = tf.random_uniform([], 
                                            -width_shift_range * im_size[1],
                                            width_shift_range * im_size[1])
          if height_shift_range:
              height_shift_range = tf.random_uniform([],
                                             -height_shift_range * im_size[0],
                                             height_shift_range * im_size[0])
          # Translate both 
          output_img = tfcontrib.image.translate(output_img,
                                                 [width_shift_range, height_shift_range])
          label_img = tfcontrib.image.translate(label_img,
                                                [width_shift_range, height_shift_range])
      return output_img, label_img

    def flip_img(self, horizontal_flip, tr_img, label_img):
        '''
        Perform horizontal flip of image for data augmentation.
        '''
        if horizontal_flip:
            flip_prob = tf.random_uniform([], 0.0, 1.0)
            tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                    lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                    lambda: (tr_img, label_img))
        return tr_img, label_img

    def _augment(self,
                 img,
                 label_img,
                 resize=None,  # Resize the image to some size e.g. [256, 256]
                 scale=1,  # Scale image e.g. 1 / 255.
                 hue_delta=0,  # Adjust the hue of an RGB image by random factor
                 horizontal_flip=False,  # Random left right flip,
                 width_shift_range=0,  # Randomly translate the image horizontally
                 height_shift_range=0):  # Randomly translate the image vertically 
        '''
        Perform data agumentation
        '''
        if resize is not None:
            # Resize both images
            label_img = tf.image.resize_images(label_img, resize)
            img = tf.image.resize_images(img, resize)
      
        if hue_delta:
            img = tf.image.random_hue(img, hue_delta)
      
        img, label_img = self.flip_img(horizontal_flip, img, label_img)
        img, label_img = self.shift_img(img, label_img, width_shift_range, height_shift_range)
        label_img = tf.to_float(label_img) * scale
        img = tf.to_float(img) * scale 
        return img, label_img
    
    def augmentation_config(self):
        '''
        Setup data augmentation params.
        '''
        im_size = self.im_size
        
        tr_cfg = {
                'resize': [im_size[0], im_size[1]],
                'scale': 1 / 255.,
                'hue_delta': 0.1,
                'horizontal_flip': True,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1
                }
        tr_preprocessing_fn = functools.partial(self._augment, **tr_cfg)

        val_cfg = {
                'resize': [im_size[0], im_size[1]],
                'scale': 1 / 255.,
                }
        val_preprocessing_fn = functools.partial(self._augment, **val_cfg)
        
#        self.tr_cfg = tr_cfg
        self.tr_preprocessing_fn = tr_preprocessing_fn
#        self.val_cfg = val_cfg
        self.val_preprocessing_fn = val_preprocessing_fn

    def get_baseline_dataset(self,
                             filenames, 
                             labels,
                             preproc_fn=functools.partial(_augment),
                             threads=5, 
                             shuffle=True):
        '''
        Create the dataset and set up batches
        '''     
        batch_size = self.batch_size
        N = len(filenames)
        
        # Create a dataset from the filenames and labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        
        # Map our preprocessing function to every element in our dataset
        dataset = dataset.map(self._process_pathnames, num_parallel_calls=threads)
        if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
            assert batch_size == 1, "Batching images must be of the same size"
    
        dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
      
        if shuffle:
            dataset = dataset.shuffle(N)
      
        # It's necessary to repeat our data for all epochs 
        dataset = dataset.repeat().batch(batch_size)
        return dataset

    def create_datasets(self):
        '''
        Create the final training dataset
        '''       
        train_images = self.train_images
        train_masks = self.train_masks
        val_images = self.val_images
        val_masks = self.val_masks
        tr_preprocessing_fn = self.tr_preprocessing_fn
        val_preprocessing_fn = self.val_preprocessing_fn
        
        train_ds = self.get_baseline_dataset(train_images,
                                             train_masks,
                                             preproc_fn=tr_preprocessing_fn)
        
        val_ds = self.get_baseline_dataset(val_images,
                                           val_masks, 
                                           preproc_fn=val_preprocessing_fn)
        return train_ds, val_ds

    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score
    
    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss
    
    def bce_dice_loss(self, y_true, y_pred):
        loss = losses.binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss
    
    def create_model(self):
        '''
        Construct the Unet architecture.
        '''
        im_size = self.im_size
        inputs = layers.Input(shape=im_size)
    
        # Encoder
        encoder0_pool, encoder0 = encoder_block(inputs, 32)
        encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
        encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
        encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
        encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
        
        # Unet center
        center = conv_block(encoder4_pool, 1024)
        
        # Decoder
        decoder4 = decoder_block(center, encoder4, 512)
        decoder3 = decoder_block(decoder4, encoder3, 256)
        decoder2 = decoder_block(decoder3, encoder2, 128)
        decoder1 = decoder_block(decoder2, encoder1, 64)
        decoder0 = decoder_block(decoder1, encoder0, 32)
        
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
        unet = models.Model(inputs=[inputs], outputs=[outputs])
        unet.compile(optimizer='adam', loss=self.bce_dice_loss, metrics=[self.dice_loss])

        self.unet = unet
        
    def train(self, plot_hist=False):
        '''
        Train the Unet.
        '''
        
        self.setup_data_paths()
        self.augmentation_config()
        train_ds, val_ds = self.create_datasets()
        
        model_name = self.model_name
        savedir = self.savedir
        
        unet = self.unet
        batch_size = self.batch_size
        epochs = self.epochs
        n_train = self.n_train
        n_val = self.n_val
        
        # Model saving and monitoring
        save_model_path = os.path.join(savedir, model_name)
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, 
                                                monitor='val_dice_loss', 
                                                save_best_only=True, 
                                                verbose=1)

        # Train
        history = unet.fit(train_ds, 
                           steps_per_epoch=int(np.ceil(n_train/float(batch_size))),
                           epochs=epochs,
                           validation_data=val_ds,
                           validation_steps=int(np.ceil(n_val/float(batch_size))),
                           callbacks=[cp])
        
        if plot_hist:
            dice = history.history['dice_loss']
            val_dice = history.history['val_dice_loss']
            
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            
            epochs_range = range(epochs)
            
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, dice, label='Training Dice Loss')
            plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Dice Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            
            plt.show()
            
        self.unet = unet
        self.hist = history
        
    def finetune(self, train_images, train_masks, val_images, val_masks, plot_hist=False):
        '''
        Train the Unet on some additional data.
        '''
        self.train_images = train_images
        self.train_masks = train_masks
        self.val_images = val_images
        self.val_masks = val_masks
        
        self.augmentation_config()
        train_ds, _ = self.create_datasets()
        
        model_name = self.model_name
        savedir = self.savedir
        
        unet = self.unet
        batch_size = self.batch_size
        epochs = self.epochs
        n_train = self.n_train
        #n_val = self.n_val
        
        # Model saving and monitoring
        save_model_path = os.path.join(savedir, model_name)
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, 
                                                monitor='val_dice_loss', 
                                                save_best_only=True, 
                                                verbose=1)

        # Train
        history = unet.fit(train_ds, 
                           steps_per_epoch=int(np.ceil(n_train/float(batch_size))),
                           epochs=epochs,
                           callbacks=[cp])
        
        if plot_hist:
            dice = history.history['dice_loss']
            val_dice = history.history['val_dice_loss']
            
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            
            epochs_range = range(epochs)
            
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, dice, label='Training Dice Loss')
            plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Dice Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            
            plt.show()
            
        self.unet = unet
        self.hist_finetune = history
        
    def test(self, test_dir=None, test_images=None):
        '''
        Uses the trained model for testing on new data
        '''
        unet = self.unet
        savedir = self.savedir
        
        if test_dir:
            test_images = list(pathlib.Path(test_dir).glob('*'))

        pred = []
        for im in test_images:
            p = unet.predict(im)[0]
            pred.append(p)
            
            savepath = os.path.join(savedir, os.path.basename(im))
            io.imsave(fname=savepath, arr=(p*255).astype(np.uint8))
            
        return pred


