import numpy as np
import wave
import struct

# Parámetros del audio
sampling_rate = 44100  # Frecuencia de muestreo
duration = 2  # Duración de cada "pum" en segundos
frequency = 150  # Frecuencia de cada "pum"
num_beats = 20  # Número de "beats"

# Generar una onda senoidal
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
audio_signal = 0.5 * np.sin(2 * np.pi * frequency * t)

# Añadir silencios entre los "beats"
silence_duration = 0.5  # Duración del silencio entre "beats"
silence = np.zeros(int(sampling_rate * silence_duration))

# Concatenar los "beats" y los silencios
beat_signal = np.concatenate([np.concatenate([audio_signal, silence]) for _ in range(num_beats)])

# Guardar el audio en un archivo WAV
output_file = 'testBeat.wav'
with wave.open(output_file, 'w') as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16 bits por muestra
    wav_file.setframerate(sampling_rate)
    wav_file.writeframes(b''.join(struct.pack('<h', int(sample * 32767)) for sample in beat_signal))

print(f'Archivo de audio guardado como {output_file}')
