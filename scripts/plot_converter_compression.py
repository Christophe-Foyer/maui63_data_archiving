# %%

import os
import tempfile

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import subprocess

import skimage.io
from contextlib import contextmanager

import numpy as np

from maui63_data_archiving.converter import convert_images

# %%

def du(path):
    "Returns size in kb"
    return subprocess.check_output(["du", "-s", path]).split()[0].decode("utf-8")


@contextmanager
def make_image_dir(num_img=20, number_of_points=50, new_points_per_iter=20, size=[800, 800], dpi=200):
    'Create a temp directory with some images'
    # Reset the figure
    plt.clf()
    with tempfile.TemporaryDirectory() as dir:
        # Initialize with N random points
        x = list(range(number_of_points))
        y = [np.random.randint(255) for _ in range(number_of_points)]
        
        for i in tqdm(range(num_img), desc="Dummy input image generation"):
            filename = f"{dir}/fig_{str(i).zfill(6)}.png"
            
            # Add M new points and drop M first points
            if i > 0:  # Skip for first frame to show initial N points
                # Drop the first M points
                x = x[new_points_per_iter:]
                y = y[new_points_per_iter:]
                
                # Add M new points (continuing the x values)
                new_x_start = x[-1] + 1 if x else 0
                x.extend(range(new_x_start, new_x_start + new_points_per_iter))
                y.extend([np.random.randint(255) for _ in range(new_points_per_iter)])
            
            plt.figure(figsize=tuple(np.array(size)/dpi), dpi=dpi)
            plt.plot(x, y)
            plt.savefig(filename, dpi=dpi)
            plt.clf()  # Clear the figure for next iteration
            
            # TODO: This is very inefficient
            skimage.io.imsave(
                filename,
                skimage.io.imread(filename)[:, :, :3]
            )
        yield dir

#  %% For frame count

# TODO: Make this work for all converters in one loop
img_counts = []
dirsize = []
filesize_h265 = []
filesize_jxl = []
num_iter = 10
iter_to_frame=lambda x: 1 + x*10

with make_image_dir(iter_to_frame(num_iter-1), number_of_points=5000, new_points_per_iter=2000) as dir:
    # Go down so we can delete extra files as we go
    for i in tqdm(reversed(range(0, num_iter)), desc="Main loop"):
        img_count = iter_to_frame(i)

        # delete extra images
        images = sorted(os.listdir(dir))
        for idx in range(img_count, len(images)):
            os.remove(os.path.join(dir, images[idx]))

        with tempfile.TemporaryDirectory() as f_h265, tempfile.TemporaryDirectory() as f_jxl:
            convert_images(dir, os.path.join(f_h265, "output.mkv"), converter_type="h265", preset="veryslow")
            convert_images(dir, f_jxl, converter_type="jxl")

            img_counts.append(img_count)
            dirsize.append(int(du(dir)))
            filesize_h265.append(int(du(f_h265)))
            filesize_jxl.append(int(du(f_jxl)))

# %%

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns = ax1.plot(img_counts, dirsize, 'b-', label="PNG size")
lns += ax1.plot(img_counts, filesize_h265, 'g-', label="H265 size")
lns += ax1.plot(img_counts, filesize_jxl, 'r-', label="JXL size")
ax1.set_xlabel("Frame Count")
ax1.set_ylabel("Size (kb)")
lns += ax2.plot(img_counts, list(np.array(filesize_h265)/np.array(dirsize) * 100), 'g:', label="H265 compression ratio")
lns += ax2.plot(img_counts, list(np.array(filesize_jxl)/np.array(dirsize) * 100), 'r:', label="JXL compression ratio")
ax2.set_ylabel("Relative Size (%)")
ax2.set_ylim(ymin=0)
ax1.legend(lns, [l.get_label() for l in lns], loc=0, facecolor='white', framealpha=0.95)
plt.title("Matplolib Figure Random Plot Panning Set - Data Size comparison")
plt.savefig("data_size_vs_frame_count.jpg")
plt.show()

# %% For image size

# TODO: Make this work for all converters in one loop
img_size = []
dirsize = []
filesize_h265 = []
filesize_jxl = []
num_iter = 10
frame_count = 30
iter_to_kwargs = lambda x: dict(
    number_of_points=500*(i+1),
    new_points_per_iter=200*(i+1),
    size=np.array([100, 100]) * (i + 1),
)

for i in tqdm(range(0, num_iter), desc="Main loop"):
    kwargs = iter_to_kwargs(i)
    with make_image_dir(frame_count, **kwargs) as dir, tempfile.TemporaryDirectory() as f_h265, tempfile.TemporaryDirectory() as f_jxl:
        convert_images(dir, os.path.join(f_h265, "output.mkv"), converter_type="h265", preset="veryslow")
        convert_images(dir, f_jxl, converter_type="jxl")

        img_size.append(kwargs["size"][0])
        dirsize.append(int(du(dir)))
        filesize_h265.append(int(du(f_h265)))
        filesize_jxl.append(int(du(f_jxl)))

# %%

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns = ax1.plot(img_size, dirsize, 'b-', label="PNG size")
lns += ax1.plot(img_size, filesize_h265, 'g-', label="H265 size")
lns += ax1.plot(img_size, filesize_jxl, 'r-', label="JXL size")
ax1.set_xlabel("Frame size")
ax1.set_ylabel("Size (kb)")
lns += ax2.plot(img_size, list(np.array(filesize_h265)/np.array(dirsize) * 100), 'g:', label="H265 compression ratio")
lns += ax2.plot(img_size, list(np.array(filesize_jxl)/np.array(dirsize) * 100), 'r:', label="JXL compression ratio")
ax2.set_ylabel("Relative Size (%)")
ax2.set_ylim(ymin=0)
ax1.legend(lns, [l.get_label() for l in lns], loc=0, facecolor='white', framealpha=0.95)
plt.title("Matplolib Figure Random Plot Panning Set - Data Size comparison")
plt.savefig("data_size_vs_frame_size.jpg")
plt.show()

# %%
