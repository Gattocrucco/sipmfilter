import numpy as np

import toy

tau = np.array([4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384])
snr = np.linspace(1.8, 6, 15)

wlen = np.array(64 * np.array([2, 2.5, 3, 4, 5, 6, 7]), int)
wlmargin = np.full_like(wlen, 64)

noise_proto0 = toy.DataCycleNoise()
noise_proto0.load('merged_000886-adc_W201_Ch00.npz')
noise_LNGS = toy.DataCycleNoise(allow_break=True)
noise_LNGS.load('nuvhd_lf_3x_tile57_77K_64V_6VoV_1-noise.npz')
noise_white = toy.WhiteNoise()
noise_obj  = [noise_proto0, noise_LNGS, noise_white]
noise_name = ['proto0',     'LNGS',     'white'    ]

template = toy.Template.load('nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz')

toys = []
for name, noise in zip(noise_name, noise_obj):
    t = toy.Toy(template, tau, snr, noise, upsampling=False)
    # upsampling=True reduces speed dramatically
    print(f'running with {name} noise...')
    t.run(100, pbar=10, seed=202012071818)
    print(f'running with {name} noise (windowed)...')
    t.run_window(wlen, wlmargin, pbar=10)
    toys.append(t)

for inoise in range(len(toys)):
    filename = f'toycomparison-{noise_name[inoise]}.npz'
    print(f'saving to {filename}...')
    toys[inoise].save(filename)
