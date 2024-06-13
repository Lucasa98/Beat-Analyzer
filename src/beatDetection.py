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

        # If rectify is specified, only return positive values
        flux = 0
    # TODO: chequear para seniales complejas
        if(len(spectrum)):
            for j in range(0,len(spectrum)):
                diff = abs(spectrum[j]) - abs(prevSpectrum[j])
                flux += diff
#        else:
#        flux = abs(spectrum) - abs(prevSpectrum)
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
file_path = './src/testBeat.wav'
sample_rate, data = wavfile.read(file_path);

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

# Part 4 - Estimate Tempo
# 4.1 Autocorrelation (formula 5 de [4])

# gaussian window
tau0 = 0.5 # seconds
sigma = 1.4 # octaves
def w(tau):
    return np.exp(-0.5*(np.log2(tau/tau0)/sigma))

# autocorrelation
def AC(tau):
    shift = int(tau*sample_rate)
    return w(tau)*np.sum(np.multiply(spectrum,np.roll(spectrum,shift)), axis=0)

# 4.2 Max TPS from 4.1.1 will be target tempo

taus = np.arange(0.1,2,0.05)
autocorrelations = [0]*len(taus)
max_ac = 0 # index
for i in range(len(taus)):
    autocorrelations[i] = AC(taus[i])
    if (autocorrelations[i] > autocorrelations[max_ac]):
        max_ac = i

estimated_tempo = taus[max_ac]
print(estimated_tempo)

# Part 5 - Peak picking based on estimated tempo

# based on estimated tempo, there must be at least n_beats beats
track_duration = n_samples/sample_rate
n_beats = track_duration * estimated_tempo

# 5.1 - calculate (1) (Beat Tracking by Dynamic Programming) for every beats combination
# single objective function
def F(dt,tau):
    return -(np.log(dt/tau)**2)
# error function C for a combination {t_i} of beats (los indices)
# TODO: definir el alfa, creo que esta en el paper
alfa = 1    # <----
def C(beats):
    sum_errors = 0
    for i in range(1,len(beats)):
        sum_errors += F(beats[i]/sample_rate,estimated_tempo)
    sum(spectrum[beats],axis=0) + alfa*sum_errors

# TODO: detectar todos los beats (agarrar todos los maximos locales y chau)
# 5.2 - pick beats from greater C in 5.1
