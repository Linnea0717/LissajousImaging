import numpy as np
import matplotlib.pyplot as plt

dir = "output/ver6/1/x512y512z31_coo/"
data = np.load(dir + 'vol_0000.npz')
Z, H, W = data['shape']
dense = np.zeros((Z, H, W), dtype=np.float32)
dense[data['z'], data['y'], data['x']] = data['signal']

for z in [0, 15, 30]:
    vmin = np.percentile(dense[z], 1)
    vmax = np.percentile(dense[z], 99.9)
    plt.imshow(dense[z], cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f'Slice {z}')
    plt.colorbar()
    plt.show()