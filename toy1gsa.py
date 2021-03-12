"""
Run a simulation at 1 GSa/s with white and LNGS noise, saving the result to
file.
"""

import numpy as np

import toy

tau = np.array([4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384]) * 8
snr = np.linspace(1.8, 6, 15)

noise_LNGS = toy.DataCycleNoise(allow_break=True, timebase=1)
noise_LNGS.load('noises/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-noise.npz')
noise_white = toy.WhiteNoise(timebase=1)
noise_obj  = [noise_LNGS, noise_white]
noise_name = ['LNGS',     'white'    ]

template = toy.Template.load('templates/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz')

toys = []
for name, noise in zip(noise_name, noise_obj):
    t = toy.Toy(template, tau, snr, noise, timebase=1)
    print(f'running with {name} noise...')
    t.run(1000, pbar=10, seed=202012081136)
    toys.append(t)

for inoise in range(len(toys)):
    filename = f'toy1gsa-{noise_name[inoise]}.npz'
    print(f'saving to {filename}...')
    toys[inoise].save(filename)
