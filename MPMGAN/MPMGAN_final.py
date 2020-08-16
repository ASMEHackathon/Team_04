# coding=utf-8

'''
Auther: Zhibo Zhang
Email: zzhang38@buffalo.edu
Date: 8/15/2020
'''

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import os
import numpy as np

class CGAN():
    def __init__(self):
        # input shape
        self.img_rows = 120
        self.img_cols = 120
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.vector_size = 3
        self.latent_dim = 500
        # Conv
        self.kernel_size = 3
        # adam
        optimizer = Adam(0.0002, 0.5)
        # D
        losses = ['binary_crossentropy']
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # G
        self.generator = self.build_generator()
        # self.generator.compile(loss='mean_squared_error', optimizer=optimizer)

        # Train G
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.vector_size,))
        img = self.generator([noise, label])
        target_label = self.discriminator([img, label])
        self.model = Model([noise, label], target_label)
        self.discriminator.trainable = False
        self.model.compile(loss=losses, optimizer=optimizer)

# %------------------------------------------------%
    def build_generator(self):
        model = Sequential()
        # 15*15*64
        model.add(Dense(10*10*64, input_dim=self.latent_dim*2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Reshape((10, 10, 64)))

        model.add(UpSampling2D(size=(3, 3)))  # 10x10 -> 30x30
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 30x30 -> 60x60
        model.add(Conv2D(32, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 60x60 -> 120x120
        model.add(Conv2D(16, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("tanh"))

        label = Input(shape=(self.vector_size,), dtype='float32', name='cc')
        label_embedding = Dense(self.latent_dim)(label)

        noise = Input(shape=(self.latent_dim,), name='dd')
        noise_layer = Dense(self.latent_dim)(label)
        model_merge = concatenate([noise_layer, label_embedding])
        model_input = Activation('tanh')(model_merge)
        img = model(model_input)
        return Model([noise, label], img)

# %------------------------------------------------%
    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=self.kernel_size, strides=2, input_shape=self.img_shape, padding="same"))  # 120x120 -> 60x60
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=2, padding="same"))  # 60x60 -> 30x30
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))  # 30x30 -> 15x15
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Flatten())
        model.add(Dense(100))

        label = Input(shape=(self.vector_size,), dtype='float32', name='aa')
        label_layer = Dense(100)(label)
        img = Input(shape=self.img_shape, name='bb')
        feature = model(img)
        merge = concatenate([feature, label_layer])
        discriminator_layer = Activation('tanh')(merge)
        discriminator_layer = Dense(1)(discriminator_layer)
        discriminator_output = Activation('sigmoid')(discriminator_layer)
        return Model([img, label], discriminator_output)

    def generate_data(self):
        # XPT data--input
        data = np.loadtxt(open("XYPT_Part01_L0001.csv", "rb"), delimiter=",", skiprows=0)
        XYP_data = np.zeros((1, 3))
        for i in range(len(data)):
            if data[i, 3] == 2:
                XYP_data = np.vstack((XYP_data, data[i, 0:3]))

        XYP_data = XYP_data[1:, :]

        print('XYP_data loaded, Shape:', np.shape(XYP_data))

        # MPM data--output
        MPM_train = np.load('MPM_train.npy')
        MPM_test = np.load('MPM_test.npy')
        MPM_data = np.vstack((MPM_train, MPM_test))
        print('MPM_data loaded, Shape:', np.shape(MPM_data))
        return XYP_data, MPM_data

    def train(self, epochs, batch_size=128, sample_interval=50):

        # load data
        XYP_data, MPM_data = self.generate_data()
        X_train = MPM_data
        y_train = XYP_data
        # Normalization
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # mean_v = (y_train.max(axis=0) + y_train.min(axis=0)) / 2
        # y_train = (y_train - mean_v) / mean_v

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # --------------------- #
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # ---------------------- #
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            idx_sample = np.random.randint(0, X_train.shape[0], batch_size)
            sampled_labels = y_train[idx_sample]

            gen_imgs = self.generator.predict([noise, sampled_labels])

            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, sampled_labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # --------------------- #
            g_loss = self.model.train_on_batch([noise, sampled_labels], np.array([1] * batch_size))

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
            epoch, d_loss[0], 100 * d_loss[1], g_loss))


            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                if not os.path.exists("MPM_DCGAN_model"):
                    os.makedirs("MPM_DCGAN_model")
                self.generator.save_weights("keras_model/G_model_without_v_final_%d.hdf5" % epoch,True)
                self.discriminator.save_weights("keras_model/D_model_without_v_final_%d.hdf5" % epoch, True)

# %-------------------- Image Generation ----------------------%
# %------------------------------------------------------------%

    def sample_images(self, epoch):
        XYP_data, MPM_data = self.generate_data()
        X_train = MPM_data
        y_train = XYP_data

        r, c = 2, 5
        noise = np.random.normal(0, 1, (5, self.latent_dim))
        # manually select several figure with different apparents
        idx_sample = [0, 1, 19, 123, 196]
        sampled_labels = y_train[idx_sample]
        sample_real = X_train[idx_sample, :, :]

        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for j in range(c):
            axs[0, j].imshow(sample_real[cnt, :, :], cmap='gray', vmin=0, vmax=255)
            axs[0, j].set_title(sampled_labels[cnt])
            axs[1, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray', vmin=0, vmax=1)
            axs[1, j].set_title(sampled_labels[cnt])
            axs[1, j].axis('off')
            cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()
# %-------------------- TEST ----------------------%
# %------------------------------------------------%
    def test(self, gen_nums=5, idx_sample = [1, 19, 123, 196, 2000]):
        XYP_data, MPM_data = self.generate_data()
        X_train = MPM_data
        y_train = XYP_data
        self.generator.load_weights("G_model_without_v000254800.hdf5", by_name=True)
        self.discriminator.load_weights("D_model_without_v000254800.hdf5", by_name=True)
        noise = np.random.normal(0, 1, (gen_nums, self.latent_dim))
        # idx_sample = np.linspace(1, 101, 100).astype(int)
        # idx_sample = [1, 19, 123, 196, 2000]
        sampled_labels = y_train[idx_sample]
        sample_real = X_train[idx_sample, :, :]


        gen = self.generator.predict([noise, sampled_labels])
        gen = 0.5 * gen + 0.5
        gen = gen.reshape(-1, 120, 120)
        print(gen.shape)
        # %------------------------------------------------%
        # visualization and save
        fig, axs = plt.subplots(2, 5)
        cnt = 0
        for j in range(len(gen)):
            axs[0, j].imshow(sample_real[j, :, :], cmap='gray', vmin=0, vmax=255)
            axs[0, j].set_title('Real'+str(j))
            axs[1, j].imshow(gen[j, :, :], cmap='gray', vmin=0, vmax=1)
            axs[1, j].set_title('Synthesis'+str(j))
            axs[0, j].axis('off')
            axs[1, j].axis('off')
            cnt += 1

        # generate one-to-one image
        # for i in range(0, len(gen)):
        #     fig, axs = plt.subplots(1, 2)
        #     axs[0].imshow(sample_real[i, :, :], cmap='gray', vmin=0, vmax=255)
        #     # axs[1, 1].set_title(sampled_labels[i])
        #     axs[1].imshow(gen[i, :, :], cmap='gray', vmin=0, vmax=1)
        #     # axs[2, 1].set_title(sampled_labels[cnt])
        #     axs[1].axis('off')


        if not os.path.exists("keras_gen"):
            os.makedirs("keras_gen")
        plt.savefig("keras_gen" + os.sep + 'example.png')
        plt.close()


if __name__ == '__main__':
    mpmgan = CGAN()
    # if not os.path.exists("./images"):
    #     os.makedirs("./images")
    # mpmgan.train(epochs=300000, batch_size=100, sample_interval=200)
    # mpmgan.test()
    # if you want, you can give any five index to generate example result
    # eg: mpmgan.test(idx_sample =[1, 19, 123, 196, 2000])
    mpmgan.test(idx_sample=[5, 19, 123, 196, 2000])