import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

import figlatex
import toy
import textbox

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'
noisefile = 'merged_000886-adc_W201_Ch00.npz'
tau = [8, 12, 16, 64, 384]
snr = np.linspace(1.8, 6, 60)
wlen = np.array(64 * np.array([1.5, 2, 2.5, 3, 4, 5, 6, 7]), int) - 32
wlmargin = np.full_like(wlen, 64 - 32)
wlmargin[0] = 16

###########################

def isomething(a, x, strict=True):
    i = min(np.searchsorted(a, x), len(a) - 1)
    assert not strict or a[i] == x
    return i
itau = lambda t: isomething(tau, t)
isnr = lambda s: isomething(snr, s, False)

simprefix = 'figwindowtempres'
names = ['proto0', 'lngs']
nicenames = dict(proto0='Proto0', lngs='LNGS')
simfile = lambda n: f'{simprefix}-{n}.npz'

if not all(os.path.exists(simfile(n)) for n in names):
    
    noise_proto0 = toy.DataCycleNoise(maxcycles=2, chunk_skip=1000)
    noise_proto0.load(noisefile)
    noise_lngs = toy.DataCycleNoise(maxcycles=2)
    noise_lngs.load(f'{prefix}-noise.npz')
    noise = dict(proto0=noise_proto0, lngs=noise_lngs)

    template = toy.Template.load(f'{prefix}-template.npz')
    
    for n in names:
        if os.path.exists(simfile(n)):
            continue
        sim = toy.Toy(template, tau, snr, noise[n])
        sim.run(1000, pbar=10, seed=202102191139)
        
        res = sim.templocres() # shape == (nfilter, ntau, nsnr)
        i = isnr(2.4)
        ifilterisnritau = np.array([
            (3, i, itau(64)), # matched N=64
            (2, i, np.argmin(res[2, :, i])), # exp, best tau
        ])
        wcenter = sim.window_center(*ifilterisnritau.T)
        
        sim.run_window(wlen, wlmargin, wcenter, pbar=10)
        
        print(f'save {simfile(n)}')
        sim.save(simfile(n))

sim = {}
for n in names:
    print(f'load {simfile(n)}')
    sim[n] = toy.Toy.load(simfile(n))
    assert np.array_equal(sim[n].tau, tau)
    assert np.array_equal(sim[n].snr, snr)

fig, axs = plt.subplots(2, 2, num='figwindowtempres', clear=True, sharex=True, sharey='row', gridspec_kw=dict(height_ratios=[2, 1]), figsize=[8.18, 7.19])

for i, n in enumerate(names):
    for j, center in enumerate([1, 2]):
        ax = axs[i, j]
        sim[n].plot_loc_window(itau=itau(384), icenter=center, logscale=False, ax=ax)
        textbox.textbox(ax, nicenames[n] + ' noise', fontsize='medium', loc='lower left')

for ax in axs.reshape(-1):
    ax.set_title(None)
    if ax.is_last_row():
        ax.set_xlabel('SNR (before filtering)')
    if ax.is_first_col():
        ax.set_ylabel('Temporal resolution [ns]')
        lims = ax.get_ylim()
        ax.set_ylim(0, lims[1])
    if ax.is_first_row() and ax.is_first_col():
        ax.legend(fontsize='small', title='Window [samples]\n$-$L+R of center', title_fontsize='small', loc='center right')
    else:
        lg = ax.legend([], [])
        lg.remove()

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
