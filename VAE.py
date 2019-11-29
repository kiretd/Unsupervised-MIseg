from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers import Lambda, Input, Dense, Layer
from keras.models import Model
from keras import losses

from skimage import morphology, io, img_as_float
from skimage.color import rgb2grey
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import norm
import pandas as pd



def nll(y_true, y_pred):
    '''
    Negative log likelihood (Bernoulli). 
    Gives the sum instead of the mean (as in Keras version).
    '''
    loss = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss

def sampling(args):
    '''
    Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    '''
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class KLDivergenceLayer(Layer):
    '''
    Identity transform layer that adds KL divergence
    to the final model loss.
    '''

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


class VAE:
    def __init__(self, loadpaths, savedir,
                 im_size=(32,32,1),
                 hidden_dim=512,
                 latent_dim=2,
                 epsilon_std=1.0,
                 batch_size=128,
                 epochs=40):
        '''
        Initialize VAE with training images in loadpaths and output images 
        sent to savedir.
        '''
        self.loadpaths = loadpaths
        self.savedir = savedir
        self.im_size = im_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.batch_size = batch_size
        self.epochs = epochs
        self.original_dim = im_size[0] * im_size[1]
        
    def load_data(self):
        '''
        Load the training data
        '''
        loadpaths = self.loadpaths
        im_size = self.im_size
        original_dim = self.original_dim
        
        Ximage = []
        for i in loadpaths:
            im = rgb2grey(img_as_float(io.imread(i)*255))
            im = resize(im, im_size, order=1, mode='reflect')
            Ximage.append(im)
        
        Ximage = np.array(Ximage).astype('float32')
        Ximage = np.reshape(Ximage, [-1, original_dim])
        return Ximage
    
    def vae_loss(self, y_true, y_pred):
        '''
        Create the VAE loss by summing reconstruction loss and KL-divergence loss.
        '''
        original_dim = self.original_dim
        z_log_var = self.z_log_var
        z_mean = self.z_mean
        
        reconstruction_loss = losses.mse(K.flatten(y_true), K.flatten(y_pred))
        reconstruction_loss *= original_dim
        
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        return K.mean(reconstruction_loss + kl_loss)
    
    def create_model(self):
        '''
        Creates the VAE architecture used for synthetic cone generation.
        '''
        original_dim = self.original_dim
        hidden_dim = self.hidden_dim
        latent_dim = self.latent_dim
        input_shape = (self.original_dim, )

        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(hidden_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        
        # use reparameterization trick to push the sampling out as input
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        
        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(hidden_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)
        
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        
        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')
        
#        # create KL loss
#        reconstruction_loss = mse(inputs, outputs)
#        reconstruction_loss *= original_dim
#        
#        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#        kl_loss = K.sum(kl_loss, axis=-1)
#        kl_loss *= -0.5
#        
#        vae_loss = K.mean(reconstruction_loss + kl_loss)
#        vae.add_loss(vae_loss)
        
        vae.compile(optimizer='adam', loss=self.vae_loss)
        
        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae
#        self.loss = vae_loss

    def train(self, plot_hist=False):
        '''
        Train the VAE.
        '''
        vae = self.vae
        batch_size = self.batch_size
        epochs = self.epochs
        
        Xtrain = self.load_data()
        
        hist = vae.fit(Xtrain,
                       Xtrain,
                       shuffle=True,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=0.20)
        
        self.vae = vae
        self.hist = hist
        
        if plot_hist:
            pd.DataFrame(hist.history).plot()

    def generate_cones(self):
        '''
        Generates and saves the required cone outlines using the trained vae.
        Uses a linearly spaced grid as noise input to create variation.
        '''
        decoder = self.decoder
        cone_size = self.im_size[0]
        
        # 50 linearly spaced points for each latent_dim dimension
        n = 50
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]
        
        outlines = []
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                
                cone = x_decoded[0].reshape(cone_size, cone_size)
                cone = cone > 0.25
                cone = morphology.convex_hull_image(cone)
                cone = (cone * 255).astype(np.uint8)
                
                cone_outline = np.zeros_like(cone)
                cone, contours, hierarchy = cv2.findContours(cone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(cone_outline, contours, -1, True, 1)
                cone_outline = morphology.skeletonize(cone_outline)
                
                outlines.append(cone_outline)
        return outlines



