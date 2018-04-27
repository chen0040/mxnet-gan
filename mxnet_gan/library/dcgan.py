import logging

import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon import nn
import os
import numpy as np
import time

from mxnet_gan.library.image_utils import load_images, save_image


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


class DCGan(object):

    model_name = 'dcgan'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.netG = None
        self.netD = None
        self.model_ctx = model_ctx
        self.data_ctx = data_ctx
        self.random_input_size = 100

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, DCGan.model_name + '-config.npy')

    @staticmethod
    def get_params_file_path(model_dir_path, net_name):
        return os.path.join(model_dir_path, DCGan.model_name + '-' + net_name + '.params')

    @staticmethod
    def create_model(num_channels=3, ngf=64, ndf=64):
        netG = nn.Sequential()
        with netG.name_scope():
            # input shape: (?, random_input_length, 1, 1)

            netG.add(nn.Conv2DTranspose(channels=ngf * 8,
                                        kernel_size=4, strides=1,
                                        padding=0, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation('relu'))
            # state shape: (?, ngf * 8, 4, 4)

            netG.add(nn.Conv2DTranspose(channels=ngf * 4,
                                        kernel_size=4, strides=2,
                                        padding=1, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation('relu'))
            # state shape: (?, ngf * 4, 8, 8)

            netG.add(nn.Conv2DTranspose(channels=ngf * 2,
                                        kernel_size=4, strides=2,
                                        padding=1, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation('relu'))
            # state shape: (?, ngf * 2, 16, 16)

            netG.add(nn.Conv2DTranspose(channels=ngf,
                                        kernel_size=4, strides=2,
                                        padding=1, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation('relu'))
            # state shape: (?, ngf, 32, 32)

            netG.add(nn.Conv2DTranspose(channels=num_channels,
                                        kernel_size=4, strides=2,
                                        padding=1, use_bias=False))
            netG.add(nn.Activation('tanh'))
            # state shape: (?, num_channels, 64, 64)

        netD = nn.Sequential()
        with netD.name_scope():
            # input shape: (?, num_channels, 64, 64)

            netD.add(nn.Conv2D(channels=ndf,
                               kernel_size=4, strides=2,
                               padding=1, use_bias=False))
            # netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            # state shape: (?, ndf, 32, 32)

            netD.add(nn.Conv2D(channels=ndf * 2,
                               kernel_size=4, strides=2,
                               padding=1, use_bias=False))
            netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            # state shape: (?, ndf * 2, 16, 16)

            netD.add(nn.Conv2D(channels=ndf * 4,
                               kernel_size=4, strides=2,
                               padding=1, use_bias=False))
            # netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            # state shape: (?, ndf * 4, 8, 8)

            netD.add(nn.Conv2D(channels=ndf * 8,
                               kernel_size=4, strides=2,
                               padding=1, use_bias=False))
            # netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            # state shape: (?, ndf * 8, 4, 4)

            netD.add(nn.Conv2D(channels=1,
                               kernel_size=4, strides=1,
                               padding=0, use_bias=False))
            # state shape: (?, 1, 1, 1)

        return netG, netD

    def load_model(self, model_dir_path):
        config = np.load(self.get_config_file_path(model_dir_path)).item()
        self.random_input_size = config['random_input_size']
        self.netG, self.netD = self.create_model()
        self.netG.load_params(self.get_params_file_path(model_dir_path, 'netG'), ctx=self.model_ctx)
        self.netD.load_params(self.get_params_file_path(model_dir_path, 'netD'), ctx=self.model_ctx)

    def checkpoint(self, model_dir_path):
        self.netG.save_params(self.get_params_file_path(model_dir_path, 'netG'))
        self.netD.save_params(self.get_params_file_path(model_dir_path, 'netD'))

    def fit(self, data_dir_path, model_dir_path, epochs=2, batch_size=64, learning_rate=0.0002, beta1=0.5,
            extension='.jpg'):

        config = dict()
        config['random_input_size'] = self.random_input_size
        np.save(self.get_config_file_path(model_dir_path), config)

        img_list = load_images(data_dir_path, extension=extension)
        img_list = [img_arr.reshape((1,) + img_arr.shape) for img_arr in img_list]
        train_data = mx.io.NDArrayIter(data=nd.concatenate(img_list), batch_size=batch_size)

        loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

        self.netG, self.netD = self.create_model()

        self.netG.initialize(mx.init.Normal(0.02), ctx=self.model_ctx)
        self.netD.initialize(mx.init.Normal(0.02), ctx=self.model_ctx)

        trainerG = gluon.Trainer(self.netG.collect_params(), 'adam', {'learning_rate': learning_rate, 'beta1': beta1})
        trainerD = gluon.Trainer(self.netD.collect_params(), 'adam', {'learning_rate': learning_rate, 'beta1': beta1})

        real_label = nd.ones((batch_size,), ctx=self.model_ctx)
        fake_label = nd.zeros((batch_size, ), ctx=self.model_ctx)

        metric = mx.metric.CustomMetric(facc)

        logging.basicConfig(level=logging.DEBUG)

        fake = []
        for epoch in range(epochs):
            tic = time.time()
            btic = time.time()
            train_data.reset()
            iter = 0
            for batch in train_data:

                # Step 1: Update netD
                data = train_data.data[0].as_in_context(self.model_ctx)
                random_input = nd.random_normal(0, 1, shape=(data.shape[0], self.random_input_size, 1, 1), ctx=self.model_ctx)

                with autograd.record():
                    # train with real image
                    output = self.netD(data).reshape((-1, 1))
                    errD_real = loss(output, real_label)
                    metric.update([real_label, ], [output, ])

                    # train with fake image
                    fake = self.netG(random_input)
                    output = self.netD(fake).reshape((-1, 1))
                    errD_fake = loss(output, fake_label)
                    errD = errD_real + errD_fake
                    errD.backward()
                    metric.update([fake_label, ], [output, ])

                trainerD.step(batch.data[0].shape[0])

                # Step 2: Update netG
                with autograd.record():
                    fake = self.netG(random_input)
                    output = self.netD(fake).reshape((-1, 1))
                    errG = loss(output, real_label)
                    errG.backward()

                trainerG.step(batch.data[0].shape[0])

                # Print log infomation every ten batches
                if iter % 10 == 0:
                    name, acc = metric.get()
                    logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                    logging.info(
                        'discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                        % (nd.mean(errD).asscalar(),
                           nd.mean(errG).asscalar(), acc, iter, epoch))
                iter = iter + 1
                btic = time.time()

            name, acc = metric.get()
            metric.reset()
            logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
            logging.info('time: %f' % (time.time() - tic))

            self.checkpoint(model_dir_path)

            # Visualize one generated image for each epoch
            fake_img = fake[0]
            fake_img = ((fake_img.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
            save_image(fake_img, os.path.join(model_dir_path, DCGan.model_name + '-training-') + str(epoch) + '.png')

    def generate(self, num_images, output_dir_path):
        for i in range(num_images):
            latent_z = nd.random_normal(loc=0, scale=1, shape=(1, self.random_input_size, 1, 1), ctx=self.model_ctx)
            img = self.netG(latent_z)[0]
            img = ((img.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
            save_image(img, os.path.join(output_dir_path, DCGan.model_name+'-generated-'+str(i) + '.png'))
