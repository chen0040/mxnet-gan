# mxnet-gan

Some GAN models that i studied while trying to learn MXNet.

* Deep Convolution GAN
* Pixel-to-Pixel GAN that performs image-to-image translation

# Usage

### Deep Convolution GAN

To run [DCGan](mxnet_gan/library/dcgan.py) using the [LFW](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz) dataset, 
run the following command:

```bash
python demo/dcgan_train.py
```

The [demo/dcgan_train.py](demo/dcgan_train.py) sample codes are shown below:

```python
import os
import sys
import mxnet as mx


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    data_dir_path = patch_path('data/lfw_dataset')
    output_dir_path = patch_path('models')

    from mxnet_gan.library.dcgan import DCGan
    from mxnet_gan.data.lfw_data_set import download_lfw_dataset_if_not_exists

    download_lfw_dataset_if_not_exists(data_dir_path)

    gan = DCGan(model_ctx=mx.gpu(0), data_ctx=mx.gpu(0))

    gan.fit(data_dir_path=data_dir_path, model_dir_path=output_dir_path)


if __name__ == '__main__':
    main()
```

The trained models will be saved into [demo/models](demo/models) folder with prefix "dcgan-*"

To run the trained models to generate new images:

```bash
python demo/dcgan_generate.py
```

The [demo/dcgan_generate.py](demo/dcgan_train.py) sample codes are shown below:

```python
import os
import sys
import mxnet as mx


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    model_dir_path = patch_path('models')
    output_dir_path = patch_path('output')

    from mxnet_gan.library.dcgan import DCGan
    gan = DCGan(model_ctx=mx.gpu(0), data_ctx=mx.gpu(0))
    gan.load_model(model_dir_path)
    gan.generate(num_images=8, output_dir_path=output_dir_path)


if __name__ == '__main__':
    main()
```

# Note

### Training with GPU

Note that the default training scripts in the [demo](demo) folder use GPU for training, therefore, you must configure your
graphic card for this (or remove the "model_ctx=mxnet.gpu(0)" in the training scripts). 


* Step 1: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (you should download CUDA® Toolkit 9.0)
* Step 2: Download and unzip the [cuDNN 7.0.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 





