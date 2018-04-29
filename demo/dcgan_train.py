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

    gan.fit(data_dir_path=data_dir_path, model_dir_path=output_dir_path, epochs=100)


if __name__ == '__main__':
    main()
