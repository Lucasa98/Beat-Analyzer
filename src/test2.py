import os
import collections
import librosa as lbs
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as sp
import numpy as np

file_path = 'src/120.mp3'

def export_to_file(onset_times):
    file_name_no_extension, _ = os.path.splitext(file_path)
    output_name = file_name_no_extension + '.beatmap.txt'
    with open(output_name, 'wt') as f:
        f.write('\n'.join(['%.4f' % onset_time for onset_time in onset_times]))

def get_onset_info(x, sr, export=0):
    # Short-time Fourier transform (STFT)
    hop_length = 512
    onset_env = lbs.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
    
    # Detect peaks in onset envelope
    onset_frames = sp.find_peaks(onset_env, height=np.mean(onset_env), distance=sr//hop_length//2)[0]
    onset_times = lbs.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    if export:
        export_to_file(onset_times)
        
    return onset_times, onset_frames

def calculate_bpm(onset_times, num_samples=10):
    if len(onset_times) < 2:
        return 0  # Not enough onsets to calculate BPM
    
    # Calculate time intervals between consecutive onsets
    intervals = np.diff(onset_times)
    
    # Take mean of intervals in chunks of num_samples
    chunk_means = []
    for i in range(0, len(intervals), num_samples):
        chunk = intervals[i:i+num_samples]
        if len(chunk) > 0:
            chunk_mean = np.mean(chunk)
            chunk_means.append(chunk_mean)
    
    # Compute the average interval across chunks
    average_interval = np.mean(chunk_means)
    
    # Convert average interval to BPM
    bpm = 60 / average_interval
    
    return bpm

def add_noise(x):
    noise = np.random.normal(0, 0.01, len(x))
    x_noisy = x + noise
    return x_noisy

def filter(x, fs, cutoff = 1000, order=9):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = sp.butter(order, normal_cutoff, btype='low', analog=False)
    y = sp.lfilter(b, a, x)
    return y

def normalize(x):
    # 1.1 Obtain min and max
    lower_bound = np.min(x)
    upper_bound = np.max(x)
    dx = upper_bound - lower_bound
    c = lower_bound / dx

    # 1.2 Normalized the data using the bounds
    normalized = (x/dx) - c
    return normalized

def visualize_onsets(x,sr):
    onset_times, onset_frames = get_onset_info(x, sr, export=1)
    
    # Calculate BPM
    bpm = calculate_bpm(onset_times)
    print(f'Average BPM: {bpm:.2f}')
    
    # Plot the waveform using librosa.display.waveshow
    plt.figure(figsize=(14, 5))
    lbs.display.waveshow(x, sr=sr, alpha=0.6)
    
    # Plot vertical lines at onset times
    for onset in onset_times:
        plt.axvline(x=onset, color='r', linestyle='--', label='Onsets' if onset == onset_times[0] else "")
    
    bpm = round(bpm, 0) if bpm > 0 else 0  # Round BPM 
    
    
    # Add labels, title, and legend
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform with Onset Times (Average BPM: {bpm})')
    plt.legend()
    plt.show()
    
def hz_to_note_name(hz):
    if hz == 0:
        return None
    note_number = lbs.hz_to_midi(hz)
    note_name = lbs.midi_to_note(note_number)
    return note_name

def get_main_key(x, sr):
    x = lbs.to_mono(x)
    
    # Perform pitch detection using the YIN algorithm
    pitches, magnitudes = lbs.core.piptrack(y=x, sr=sr)
    
    # Extract the pitch frequencies
    pitch_frequencies = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_frequencies.append(pitch)

    # Convert detected pitch frequencies to note names
    note_names = [hz_to_note_name(f) for f in pitch_frequencies if hz_to_note_name(f) is not None]
    avg_note = sum(pitch_frequencies) / len(pitch_frequencies)
    avg_note_name = hz_to_note_name(avg_note)
    return avg_note_name
    
def main():
    # Load the audio file
    x, sr = lbs.load(file_path)
    main_key = get_main_key(x,sr)
    # Add noise to the signal
    x_noisy = add_noise(x)
    # Filter the signal
    x_noisy = filter(x_noisy,sr)
    # Normalize the signal
    x_noisy = normalize(x_noisy)
    # Visualize onsets
    visualize_onsets(x_noisy,sr)
    
    print('Avg key:', main_key)

    # sd.play(x_noisy, sr)
    # sd.wait()

main()