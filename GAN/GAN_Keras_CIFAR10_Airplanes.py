# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:01:50 2019

@author: Joseph.Shingleton
"""
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.datasets.cifar10 import load_data

import matplotlib.pyplot as plt

import sys

import numpy as np

from Data_split import seperate_cifar10_classes, recombine_cifar10_classes

cifar10_data = seperate_cifar10_classes()
data = cifar10_data[0]
data_arr = np.asarray(data)
data_flt = data_arr.astype('float32')
data_nrm = (data_flt - 127.5)/127.5
#def load_real_samples():
#    # load cifar10 dataset
#    (trainX, _), (_, _) = load_data()
#    # convert from unsigned ints to floats
#    X = trainX.astype('float32')
#    # scale from [0,255] to [-1,1]
#    X = (X - 127.5) / 127.5
#    return X
    

#for ii in range(1,6):
#    data = load_cifar10_batch(ii, 'cifar-10-batches-py') 

class GAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)
        latent_dim = 100

        model = Sequential()
        
        model = Sequential()
        # foundation for 4x4 image
        n_nodes = 256 * 4 * 4
        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))
        # upsample to 8x8
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 16x16
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 32x32
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # output layer
        model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))


#        model.add(Dense(256, input_shape=noise_shape))
#        model.add(LeakyReLU(alpha=0.2))
#        model.add(BatchNormalization(momentum=0.8))
#        model.add(Dense(512))
#        model.add(LeakyReLU(alpha=0.2))
#        model.add(BatchNormalization(momentum=0.8))
#        model.add(Dense(1024))
#        model.add(LeakyReLU(alpha=0.2))
#        model.add(BatchNormalization(momentum=0.8))
#        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
#        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()
        
        model.add(Conv2D(64, (3,3), padding='same', input_shape=img_shape))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        
        
        
#        model.add(Flatten(input_shape=img_shape))
#        model.add(Dense(512))
#        model.add(LeakyReLU(alpha=0.2))
#        model.add(Dense(256))
#        model.add(LeakyReLU(alpha=0.2))
#        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)
    
    
    def train(self, epochs, batch_size=128, save_interval=50):

        
        X_train = data_nrm

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, np.shape(X_train)[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("CIFAR10_gen_imgs\Airplane_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, save_interval=200)