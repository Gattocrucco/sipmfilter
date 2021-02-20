"""
Run simulations at 125 MSa/s with white, LNGS and Proto0 noise, with filtering
in windows, and save the results to file. The arrays hardcoded at the beginning
of the script set the various parameters.
"""

import numpy as np

import toy

tau = np.array([4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384])
snr = np.linspace(1.8, 6, 15)

wlen = np.array(64 * np.array([1.5, 2, 2.5, 3, 4, 5, 6, 7]), int) - 32
wlmargin = np.full_like(wlen, 64 - 32)
wlmargin[0] = 16

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
    t.run(1000, pbar=10, seed=202012071818)
    
    # Make window center argument for Toy.run_window.
    isnr = np.searchsorted(snr, 2.4)
    res = t.templocres() # shape == (nfilter, ntau, nsnr)
    ifilterisnritau = np.array([
        (3, -1, -1), # matched, high snr, high tau
        (3, isnr, 7), # matched, tau=64
        (2, isnr, np.argmin(res[2, :, isnr])), # exp, best tau
        (1, isnr, np.argmin(res[1, :, isnr])), # movavg, best tau
    ])
    wcenter = t.window_center(*ifilterisnritau.T)
    
    print(f'running with {name} noise (windowed)...')
    t.run_window(wlen, wlmargin, wcenter, pbar=10)
    toys.append(t)

for inoise in range(len(toys)):
    filename = f'toycomparison-{noise_name[inoise]}.npz'
    print(f'saving to {filename}...')
    toys[inoise].save(filename)
