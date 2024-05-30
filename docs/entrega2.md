# Problemas

## Beat Tracking

Consiste en identificar los BPM (Beats Per Minute o Pulsaciones Por Minuto) de un track (canción)

## Music Note and Pitch detection

Identificar la nota y el tono de la nota predominante

# Propuestas

## Beat Tracking

En el caso más elemental tendremos un tono que suena una determinada cantidad de veces por segundo (Hz), lo cual es fácilmente convertible a BPM (Hz\*60). Si realizamos la transformadda de fourier de este tono, tendremos la frecuencia del tono (no lo BPM). Para contar los bpm (veces aparece el tono por minuto) tendremos
