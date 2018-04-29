import os
import sys
import mxnet as mx
from random import shuffle
import numpy as np


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    model_dir_path = patch_path('models')

    from mxnet_gan.library.pixel2pixel import Pixel2PixelGan
    from mxnet_gan.data.facades_data_set import load_image_pairs
    from mxnet_gan.library.image_utils import load_image, visualize

    img_pairs = load_image_pairs(patch_path('data/facades'))

    gan = Pixel2PixelGan(model_ctx=mx.gpu(0), data_ctx=mx.gpu(0))
    gan.load_model(model_dir_path)

    shuffle(img_pairs)

    for source_img_path, _ in img_pairs[:20]:
        source_img = load_image(source_img_path, gan.img_width, gan.img_height)
        target_img = gan.generate(source_image=source_img)
        img = mx.nd.concat(source_img.as_in_context(gan.model_ctx), target_img, dim=2)
        # img = ((img.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
        visualize(img)


if __name__ == '__main__':
    main()
