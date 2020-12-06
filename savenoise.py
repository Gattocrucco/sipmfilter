"""
Load noise data from unversioned files and write it into versioned files, then
check that loading works.
"""

import toy
from matplotlib import pyplot as plt

fig = plt.figure('savenoise')
fig.clf()

axs = fig.subplots(3, 1)

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'
source = f'{prefix}.wav'
dest = f'{prefix}-noise.npz'
print(f'saving {source} to {dest}...')
lngs1GSas = toy.DataCycleNoise(timebase=1)
lngs1GSas.load_LNGS_wav(source, 1000)
lngs1GSas.save(dest)

lngs1GSas = toy.DataCycleNoise(timebase=1)
lngs1GSas.load(dest)
axs[0].set_title('LNGS 1 GSa/s')
axs[0].plot(lngs1GSas.generate(1, 1000)[0])

lngs125MSas = toy.DataCycleNoise(timebase=8)
lngs125MSas.load(dest)
axs[1].set_title('LNGS 125 MSa/s')
axs[1].plot(lngs125MSas.generate(1, 1000)[0])

prefix = 'merged_000886'
channel = 'adc_W201_Ch00'
source = f'{prefix}.root'
dest = f'{prefix}-{channel}.npz'
print(f'saving {source} to {dest}...')
proto0125MSas = toy.DataCycleNoise(timebase=8)
proto0125MSas.load_proto0_root_file(source, channel, 150)
proto0125MSas.save(dest)

proto0125MSas = toy.DataCycleNoise(timebase=8)
proto0125MSas.load(dest)
axs[2].set_title('Proto0 125 MSa/s')
axs[2].plot(proto0125MSas.generate(1, 1000)[0])

fig.tight_layout()
fig.show()
