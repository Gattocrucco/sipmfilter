"""
Extract noise data from all LNGS wavs with the name satisfying the pattern
'LF_TILE15_77K_*VoV_1.wav' and save them as `toy.DataCycleNoise` objects.
"""

import glob

import toy
from matplotlib import pyplot as plt

fig = plt.figure('savenoise2')
fig.clf()

axs = fig.subplots(2, 1)
axs[0].set_title('LNGS 1 GSa/s')
axs[1].set_title('LNGS 125 MSa/s')

sources = list(sorted(glob.glob('LF_TILE15_77K_*VoV_1.wav')))

for source in sources:
    suffix = '.wav'
    dest = source[:-len(suffix)] + '-noise.npz'
    print(f'saving {source} to {dest}...')
    lngs1GSas = toy.DataCycleNoise(timebase=1)
    lngs1GSas.load_LNGS_wav(source, 1100)
    lngs1GSas.save(dest)

    lngs1GSas = toy.DataCycleNoise(timebase=1)
    lngs1GSas.load(dest)
    axs[0].plot(lngs1GSas.generate(1, 1000)[0])

    lngs125MSas = toy.DataCycleNoise(timebase=8)
    lngs125MSas.load(dest)
    axs[1].plot(lngs125MSas.generate(1, 1000)[0])

fig.tight_layout()
fig.show()
