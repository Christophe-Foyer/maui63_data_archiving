import os
import tempfile

import imageio.v3 as iio
import numpy as np

from maui63_data_archiving.image_sequence_to_video import images_to_lossless_h265


def test_images_to_lossless_h265():
    with tempfile.NamedTemporaryFile(suffix=".mkv") as f:
        # TODO: Use non temp files
        dir = "tmp/imagedir"
        images = [os.path.join(dir, file) for file in os.listdir(dir)]

        images_to_lossless_h265(images, f.name)

        for idx, image_file in enumerate(images):
            image = iio.imread(image_file)

            # read a single frame
            frame = iio.imread(
                f.name,
                index=idx,
            )

            # Check
            np.testing.assert_array_equal(image, frame)
