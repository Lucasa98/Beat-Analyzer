import os
import collections
import librosa as lbs
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as sp
import numpy as np

file_path = 'src/testBeat.wav'

def export_to_file(onset_times):
    file_name_no_extension, _ = os.path.splitext(file_path)
    output_name = file_name_no_extension + '.beatmap.txt'
    with open(output_name, 'wt') as f:
        f.write('\n'.join(['%.4f' % onset_time for onset_time in onset_times]))

def get_onset_info(x, sr, export=0):
    """ Obtener picos en el envolvente de la señal

    Args:
        x: señal
        sr: frecuencia de muestreo
        export: exportar valores en un archivo. Por defecto falso

    Returns:
        onset_times: instantes de tiempo de los beats
        onset_frames: posicion de los beats en el muestreo
    """
    hop_length = 512
    # envolvente de la señal a partir de Short Time Fourier Transform
    onset_env = lbs.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)

    # Detectar picos en la envolvente de la señal (posiciones en el muestreo)
    onset_frames = sp.find_peaks(onset_env, height=np.mean(onset_env), distance=sr//hop_length//2)[0]
    # Transformar de posiciones en muestreo a instantes de tiempo
    onset_times = lbs.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # exportar archivo con tiempos de beats (o no)
    if export:
        export_to_file(onset_times)

    return onset_times, onset_frames, onset_env

def calculate_bpm(onset_times, num_samples=10):
    """ Estimar BPM a partir de una lista de tiempos beats

    Args:
        onset_times: instantes de tiempo donde hay beats
        num_samples: numero de beats a tomar para calcular el promedio

    Returns: tempo estimado
    """
    # si hay muy pocos beats, nada
    if len(onset_times) < 2:
        return 0

    # Calcular intervalo entre beats consecutivos
    intervals = np.diff(onset_times)

    # Calcular intervalo promedio en trozos de num_samples muestas
    chunk_means = []
    for i in range(0, len(intervals), num_samples):
        chunk = intervals[i:i+num_samples]
        if len(chunk) > 0:
            chunk_mean = np.mean(chunk)
            chunk_means.append(chunk_mean)

    # Calcular el intervalo promedio entre los promedios de los trozos
    average_interval = np.mean(chunk_means)

    # Convertir de segundos a BPM
    bpm = 60 / average_interval

    return bpm

def add_noise(x):
    noise = np.random.normal(0, 0.01, len(x))
    x_noisy = x + noise
    return x_noisy

def filter(x, fs, cutoff = 500, order=9):
    """ Filtro Butterworth pasa bajo

    Args:
        x: señal
        fs: frecuencia de muestreo
        cutoff: señal de filtrado. Por defecto 500
        order: orden del filtro diseñado

    Returns: señal filtrada
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # funcion para diseñar filtro
    b, a = sp.butter(order, normal_cutoff, btype='low', analog=False)
    # aplicar filtro
    y = sp.lfilter(b, a, x)
    return y

def normalize(x):
    """ normaliza la señal dejandola con valores no-negativos

    Args:
        x: señal
    Returns: señal normalizada
    """
    # Obtener minimo y maximo
    lower_bound = np.min(x)
    upper_bound = np.max(x)
    dx = upper_bound - lower_bound
    # variable auxiliar para desplazar la señal normalizada y el minimo sea cero
    c = lower_bound / dx

    # normalizar utilizando el minimo y el maximo
    normalized = (x/dx) - c
    return normalized

def visualize_onsets(x,sr):
    """ Visualizar beats que definen el tempo de la pista

    Args:
        x: señal
        sr: frecuencia de muestreo
    """
    # obtener tiempos de los beats
    onset_times, onset_frames, onset_env = get_onset_info(x, sr, export=1)

    # Estimar BPM
    bpm = calculate_bpm(onset_times)
    print(f'Average BPM: {bpm:.2f}')

    # Graficar sonograma de la señal
    plt.figure(figsize=(14, 5))
    lbs.display.waveshow(x, sr=sr, alpha=0.6)
    lbs.display.waveshow(onset_env, alpha=0.6)

    # Marcar beats que marcan el tempo
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

    # Usar el algoritmos de yin para detectas los tonos
    pitches, magnitudes = lbs.core.piptrack(y=x, sr=sr)

    # Extrar las frecuencias de los tonos
    pitch_frequencies = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_frequencies.append(pitch)

    # Convertir las frecuencias detectadas a nombres de notas
    note_names = [hz_to_note_name(f) for f in pitch_frequencies if hz_to_note_name(f) is not None]

    # Calcular tono promedio (en frecuencia)
    avg_note = sum(pitch_frequencies) / len(pitch_frequencies)
    # Convertir frecuencia promedio a nombre de la nota
    avg_note_name = hz_to_note_name(avg_note)

    return avg_note_name

def main():
    # Generación de la Señal de prueba
    ## Cargar archivo
    x, sr = lbs.load(file_path)
    ## Añadir ruido
    x_noisy = add_noise(x)

    # Procesamiento de la señal
    ## Detección de beats e Identificación de tempo
    ### Filtro butterworth pasa bajo
    x_noisy = filter(x_noisy,sr)
    ### Normalizar la señal
    x_noisy = normalize(x_noisy)
    ### Visualizar beats
    visualize_onsets(x_noisy,sr)

    ## Identificación de tonalidad
    main_key = get_main_key(x,sr)
    print('Avg key:', main_key)

    # reproducir audio ruidoso
    # sd.play(x_noisy, sr)
    # sd.wait()

main()