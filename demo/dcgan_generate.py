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
