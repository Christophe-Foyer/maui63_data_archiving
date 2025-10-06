import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt

import subprocess

# import png
import skimage.io
from contextlib import contextmanager

import numpy as np
import pytest

from maui63_data_archiving.converter import convert_images


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

            # Append a bunch of data points
            for i in range(10):
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


@pytest.mark.parametrize("converter_type", ["h265", "jxl"])
def test_images_to_lossless(converter_type):
    with make_image_dir() as dir, tempfile.TemporaryDirectory() as f:
        output_path = os.path.join(f, "output.mkv")

        convert_images(dir, output_path, converter_type=converter_type, check_frames=True)


@pytest.mark.parametrize("converter_type", ["h265", "jxl"])
def test_images_to_lossless_smaller(converter_type):
    with make_image_dir() as dir, tempfile.TemporaryDirectory() as f:
        output_path = os.path.join(f, "output.mkv")
        
        convert_images(dir, output_path, converter_type=converter_type, check_frames=False)

        assert du(dir) > du(f), f"Output was larger than the input ({du(dir)} vs {du(f)})"
