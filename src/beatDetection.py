import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import scipy as scy
import envelope
from scipy.io import wavfile

def butter_lowpass_filter(data, fps, cutoff = 10, order = 2):
    nyq = 0.5 * fps
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = scy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scy.signal.filtfilt(b, a, data)
    return y

"""
SpectralFlux.py
Compute the spectral flux between consecutive spectra
This technique can be for onset detection
"""
def spectral_flux(s):
    """
    Compute the spectral flux between consecutive spectra
    """
    # flux for zeroth spectrum
    spectralFlux = [0]

    # Compute flux for subsequent spectra
    for i in range(1, len(s)):
        prevSpectrum = s[i - 1]
        spectrum = s[i]

        flux = abs(spectrum) - abs(prevSpectrum)

        # If rectify is specified, only return positive values
        if flux < 0:
            flux = 0

        spectralFlux.append(flux)

    return spectralFlux

# Beat Detection Steps
#  0. Load sound
#  1. Normalize
#  2. Filter
#  3. Apply envelope
#  4. Find peaks
#  5. Find beat based on the peaks

# Part 0 - Load sound

# 0.1 Load sound
file_path = './testBeat.wav'
data, sample_rate = envelope.read_wav(file_path);

print("samples: ", len(data))

n_samples = len(data)
white_noise = np.random.normal(0, 1, n_samples)

#data = data + white_noise

# 0.2 Plot the sound
# plt.plot(data) 
# plt.show()

# 0.3 Play Sound
# sd.play(data, sample_rate)
# sd.wait()

# Part 1 - Normalize

# 1.1 Obtain min and max
lower_bound = np.min(data)
upper_bound = np.max(data)

# 1.2 Normalized the data using the bounds
#normalized = (data - lower_bound) / (upper_bound - lower_bound)
normalized = data

# 1.3 Plot the normalized data
# plt.plot(normalized)
# plt.show()

# Part 2 - Filter

# 2.1 Define filter parameters
frequency = 1000;
w0 = frequency * np.pi;   # half the frequency as passband
num = w0;
den = [1,w0];

# 2.2 Generate transfer function for butterworth filter
lowPass = scy.signal.TransferFunction(num, den);
discrete_lowpass = lowPass.to_discrete(num, method = 'gbt',alpha = 0.5);

a = discrete_lowpass.num[1:]
b = -discrete_lowpass.den

# print("Filter coefficients: ", a, b);

# 2.3 Apply filter

filtered = scy.signal.lfilter(a, b, data)

# 2.4 Plot the filtered data

# plt.plot(filtered)
# plt.show()

# Part 3 - Detect onset

# 2.1 Spectral flux

spectrum = spectral_flux(filtered)

plt.figure(figsize=(12, 6))
plt.plot(filtered, label='Normalized Signal')
plt.plot(spectrum, label='Envelope', linewidth=2)
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Signal and its Envelope')
plt.show()

# Part 4 - Peak picking
# 4.1 Estimate Tempo
# 4.1.1 Autocorrelation (formula 5 de [4])
# 4.1.2 Max TPS from 4.1.1 will be target tempo

# Part 5 - Find beat based on the peaks