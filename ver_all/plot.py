import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

vol1 = tiff.imread(
    "output/1/volumes_z31_weighted/volume_00000.tiff"
)

vol2 = tiff.imread(
    "20251229data/bi_d/xytz_m1/1/z_000.tiff"
)

for i in range(5):
    img1 = vol1[i]
    img2 = vol2[i]

    vmin = np.percentile(img1, 1)
    vmax = np.percentile(img1, 99.9)
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap="gray", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("From current reconstruction code")

    vmin2 = np.percentile(img2, 1)
    vmax2 = np.percentile(img2, 99.9)
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap="gray", vmin=vmin2, vmax=vmax2)
    plt.colorbar()
    plt.title("From previous code")
    plt.show()