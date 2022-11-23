## Python Dependencies

-   numpy
-   matplotlib
-   soundfile

## Features

When designing wavetables, it's helpful to both view/analyse and listen to your signals.

-   Run `python view.py` to generate an interactive graph like the one below. Notice the horizontal slider below with the label 'Wavetable'.

    ![matplotlib_wavetable](/view_example.png?raw=true 'matplotlib_wavetable')

-   Run `python export.py` to generate an audio file from a selection of different formats (.flac, .wav, .aiff, .ogg)

## Algorithms

All the alogithms are in `./src/methods.py`. Change the method within the `create_wave_function` lambda function found at the beginning of `./src/methods.py` to change the wave displayed by `./view.py` & `./export.py`.
