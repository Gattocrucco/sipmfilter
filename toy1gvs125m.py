"""
Run simulations at various sampling frequencies, saving the results to file.
Usage:

    toy1gvs125m.py [<LNGS wav>.wav]

The LNGS wav is used for the noise and the template, which must have already
been saved to separate files with `savetemplate.py` and `savenoise.py` or
`savenoise2.py`. If not specified, it is set to
'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav', for which the noise and
template files are committed in the repository so there's no need to download
the wav file.

The `timebase` array hardcoded near the beginning of this script sets the
sampling frequencies used (the "timebase" is the sample step in nanoseconds).
The `tau` and `snr` arrays are the range of filter length and SNR simulated.
"""

import sys
import os

import numpy as np

import toy
import template as _template

if len(sys.argv) == 1:
    source = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
else:
    source = sys.argv[1]
suffix = '.wav'
assert source.endswith(suffix)
prefix = os.path.split(source)[1][:-len(suffix)]

timebase = [1, 8, 16, 32] # keep 1 first

tau = np.array([2048])
snr_1GSa = 0.6 * np.logspace(0, 1, 50)

##########################################################

assert timebase[0] == 1

templfile = f'templates/{prefix}-template.npz'
print(f'read {templfile}...')
template = _template.Template.load(templfile)

savedir = 'toy1gvs125m'
os.makedirs(savedir, exist_ok=True)

for whitenoise in [True, False]:

    if whitenoise:
        noise = [toy.WhiteNoise(timebase=t) for t in timebase]
        noise_name = 'white'
    else:
        noise_file = f'noises/{prefix}-noise.npz'
        print(f'read {noise_file}...')
        noise = []
        for i in range(len(timebase)):
            n = toy.DataCycleNoise(maxcycles=2, timebase=timebase[i])
            n.load(noise_file)
            noise.append(n)
        noise_name = 'lngs'

    for i in range(len(timebase)):
        tb = timebase[i]
        noise_ratio = np.std(toy.downsample(noise[0].generate(100, 100), tb), axis=None)
        snr = snr_1GSa / noise_ratio
        t = toy.Toy(template, tau // tb, snr, noise[i], timebase=tb)
        print(f'running with timebase={tb} (SNR *= {1/noise_ratio:.2g}), {noise_name} noise...')
        t.run(1000, pbar=10, seed=202012111417)
        filename = f'{savedir}/toy1gvs125m-{prefix}-{timebase[i]}-{noise_name}.npz'
        print(f'saving to {filename}...')
        t.save(filename)
