# Bibliografia de referencia

<div id="[1]">[1] McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.</div>
<div id="[2]">[2] Ellis, Daniel PW. "Beat tracking by dynamic programming." Journal of New Music Research.</div>
<div id="[3]">[3] Böck, Sebastian, Florian Krebs, and Markus Schedl. "Evaluating the Online Capabilities of Onset Detection Methods." ISMIR. 2012.</div>

# Extractos

## Onsets, tempo and beats <sup style="font-size: 12px">[[1](#[1])]</sup>

The onset module provides two functions: _onset_strength_ and _onset_detect_.
**onset_strength** function calculates a thresholded spectral flux operation over a spectrogram, and returns a one dimensional array representing the amount of increasing spectral energy at each frame.
**onset_detect** function, on the other hand, selects peak positions from the onset strength curve following the heuristic described by Boeck et al. [[3](#[3])].
The beat module provides functions to estimate the _global tempo_ and _positions of beat_ events from the onset strength function, using the method of Ellis [[2](#[2])]. More specifically, the beat tracker first estimates the tempo, which is then used to set the target spacing between peaks in an onset strength function.
