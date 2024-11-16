# %%

import os

import imageio.v2 as iio2
import imageio.v3 as iio3
import numpy as np
import skimage.io
from tqdm.autonotebook import tqdm

x265_presets = [
    "placebo",
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]


def images_to_lossless_h265(
    image_paths, output_file, check_frames=True, preset="veryslow"
):
    assert (
        preset in x265_presets
    ), f"Preset must be one of {x265_presets} but got: {preset}"

    # Setup the writer with flags
    w = iio2.get_writer(
        output_file,
        format="FFMPEG",
        mode="I",
        fps=1,
        codec="libx265",
        output_params=["-preset", preset, '-x265-params', 'lossless=1'],
        pixelformat="gbrp",
    )

    # Write the frames to the file
    for image_path in tqdm(image_paths, desc="Writing images..."):
        w.append_data(skimage.io.imread(image_path))

    w.close()

    if check_frames:
        for idx, image_path in enumerate(tqdm(image_paths, desc="Checking images...")):
            # Read a single frame
            frame = iio3.imread(
                output_file,
                index=idx,
            )

            # Check
            image = skimage.io.imread(image_path)
            np.testing.assert_array_equal(image, frame)


# %%

if __name__ == "__main__":
    import os

    import imageio.v2 as iio2
    import skimage.io

    # dir = "tmp/imagedir"
    dir = os.path.expanduser("~/Downloads/maui63_images")
    images = [os.path.join(dir, file) for file in os.listdir(dir)]

    images_to_lossless_h265(images, "tmp/output.mkv", preset='veryslow', check_frames=True)

# %%
