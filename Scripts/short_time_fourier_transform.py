import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

df = pd.read_csv("Datasets/synthetic_micro_doppler_dataset.csv")

X = df.iloc[:, :-1].values.astype(float)
y = df['label'].values

sample_bird = X[y == 0][0] 
sample_drone = X[y == 1][0]  

fs = 1000   # sampling frequency
windor_size = 128  
overlap_windows = 120

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# STFT for Class 0 (Bird)
f0, t0, Zxx0 = signal.stft(sample_bird, fs=fs, nperseg=windor_size, noverlap=overlap_windows)
mesh_0 = axes[0].pcolormesh(t0, f0, np.abs(Zxx0), vmin=0, vmax=np.abs(Zxx0).max(), shading='gouraud', cmap='viridis')
axes[0].set_title('Micro-Doppler Spectrogram (Label 0 - Bird)')
axes[0].set_ylabel('Frequency [Hz]')
axes[0].set_xlabel('Time [sec]')
fig.colorbar(mesh_0, ax=axes[0], label='Magnitude')

# STFT for Class 1 (Drone)
f1, t1, Zxx1 = signal.stft(sample_drone, fs=fs, nperseg=windor_size, noverlap=overlap_windows)
mesh_1 = axes[1].pcolormesh(t1, f1, np.abs(Zxx1), vmin=0, vmax=np.abs(Zxx1).max(), shading='gouraud', cmap='viridis')
axes[1].set_title('Micro-Doppler Spectrogram (Label 1 - Drone)')
axes[1].set_ylabel('Frequency [Hz]')
axes[1].set_xlabel('Time [sec]')
fig.colorbar(mesh_1, ax=axes[1], label='Magnitude')

plt.tight_layout()
plt.show()