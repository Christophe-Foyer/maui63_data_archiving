import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt

import subprocess

# import png
import skimage.io
from contextlib import contextmanager

import numpy as np

from maui63_data_archiving.image_sequence_to_video import images_to_lossless_h265

matplotlib.use("Agg")


def du(path):
    "Returns size in kb"
    return subprocess.check_output(["du", "-s", path]).split()[0].decode("utf-8")


# TODO: Make this a fixture
@contextmanager
def make_image_dir(num_img=20):
    'Create a temp directory with some images'
    with tempfile.TemporaryDirectory() as dir:
        x = []
        y = []
        for i in range(num_img):
            filename = f"{dir}/fig_{str(i).zfill(6)}.png"
            x.append(i)
            y.append(np.random.randint(255))
            plt.plot(x, y)
            plt.savefig(filename)
            # TODO: This is very inefficient 
            skimage.io.imsave(
                filename,
                skimage.io.imread(filename)[:, :, :3]
            )

        yield dir


def test_images_to_lossless_h265():
    with make_image_dir() as dir, tempfile.NamedTemporaryFile(suffix=".mkv") as f:
        images = [os.path.join(dir, file) for file in os.listdir(dir)]

        images_to_lossless_h265(images, f.name, check_frames=True)


def test_images_to_lossless_h265_smaller():
    with make_image_dir() as dir, tempfile.NamedTemporaryFile(suffix=".mkv") as f:
        images = [os.path.join(dir, file) for file in os.listdir(dir)]

        images_to_lossless_h265(images, f.name, check_frames=False)

        assert du(dir) > du(f.name)
