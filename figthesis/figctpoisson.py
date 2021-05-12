import os

from matplotlib import pyplot as plt
import gvar

import figlatex
import afterpulse_tile21
import textbox

cache = 'figthesis/figctpoisson.gvar'

################

gvar.switch_gvar()

if not os.path.exists(cache):
    vovdict = {}
    for vov in afterpulse_tile21.AfterPulseTile21.defaultparams:
        ap21 = afterpulse_tile21.AfterPulseTile21(vov)
        for overflow in [True, False]:
            for fixzero in [False, True]:
                _, fig1, fig2 = ap21.maindict(overflow=overflow, fixzero=fixzero)
                plt.close(fig1)
                plt.close(fig2)
        vovdict[vov] = ap21.results
    with open(cache, 'wb') as file:
        print(f'write {cache}...')
        gvar.dump(vovdict, file)

with open(cache, 'rb') as file:
    print(f'read {cache}...')
    vovdict = gvar.load(file)

fig, axs = plt.subplots(1, 4, num='figctpoisson', clear=True, figsize=[9, 3])

plotter = afterpulse_tile21.Plotter(vovdict)

model_params = [
    ('Borel', ''    , '#000'),
    ('Geom.', 'geom', '#f55'),
]

config = sum([
    [
        (f'{name} OF', plotter.paramgetter(f'mainfitof{infix}'  , 'mu_poisson'), dict(color=color, marker='o')),
        ('No OF'     , plotter.paramgetter(f'mainfit{infix}'    , 'mu_poisson'), dict(color=color, marker='o', markerfacecolor='#fff')),
        ('OF, FZ'    , plotter.paramgetter(f'mainfitof{infix}fz', 'mu_poisson'), dict(color=color, marker='s')),
        ('No OF, FZ' , plotter.paramgetter(f'mainfit{infix}fz'  , 'mu_poisson'), dict(color=color, marker='s', markerfacecolor='#fff')),
    ]
    for name, infix, color in model_params
], [])

for ivov, (vov, results) in enumerate(vovdict.items()):
    for i, (_, getter, style) in enumerate(config):
        param = getter(results)
        n = len(config) - 1
        offset = 0.075 * (i - n/2)
        axs[ivov].errorbar(gvar.mean(param), i, xerr=gvar.sdev(param), capsize=4, **style)

plotter.plotmupoisson(axs[-1])

for ivov, vov in enumerate(vovdict):
    textbox.textbox(axs[ivov], f'{vov} VoV', fontsize='medium', loc='lower right')

for ax in axs[:3]:
    ax.set_yticks(list(range(len(config))))
    ax.set_ylim(len(config), -1)
    if ax.is_first_col():
        ax.set_yticklabels([label for label, _, _ in config])
    else:
        ax.set_yticklabels([])
    ax.set_xlabel('$\\mu_P$')
    ax.minorticks_on()
    ax.set_yticks([], minor=True)
    ax.grid(which='major', axis='x', linestyle='--')
    ax.grid(which='minor', axis='x', linestyle=':')

ax = axs[-1]
ax.set_xlabel('Overvoltage [V]')
ax.set_ylabel('$\\mu_P$')

fig.tight_layout()
fig.show()

figlatex.save(fig)
