import numpy as np

import toy

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'

timebase = [1, 8, 16, 32]

whitenoise = True

if whitenoise:
    noise = [toy.WhiteNoise(timebase=t) for t in timebase]
    noise_name = 'white'
else:
    noise_file = f'{prefix}-noise.npz'
    noise = []
    for i in range(len(timebase)):
        n = toy.DataCycleNoise(allow_break=True, timebase=timebase[i])
        n.load(noise_file)
        noise.append(n)
    noise_name = 'lngs'

template = toy.Template.load(f'{prefix}-template.npz')

tau = np.array([256, 512, 1024, 2048])
snr = 0.6 * np.logspace(0, 1, 50)

toys = []
for i in range(len(timebase)):
    tb = timebase[i]
    t = toy.Toy(template, tau // tb, snr, noise[i], timebase=tb)
    print(f'running with timebase={tb}...')
    t.run(1000, pbar=10, seed=202012111417) # 202012091630
    toys.append(t)

for i in range(len(toys)):
    filename = f'toy1gvs125m-{timebase[i]}-{noise_name}.npz'
    print(f'saving to {filename}...')
    toys[i].save(filename)
