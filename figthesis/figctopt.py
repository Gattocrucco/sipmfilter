import os

from matplotlib import pyplot as plt
import gvar

import figlatex
import afterpulse_tile21
import textbox

cache = 'figthesis/figctopt.gvar'

################

gvar.switch_gvar()

if not os.path.exists(cache):
    vovdict = {}
    for vov in afterpulse_tile21.AfterPulseTile21.defaultparams:
        ap21 = afterpulse_tile21.AfterPulseTile21(vov)
        for overflow in [True, False]:
            _, fig1, fig2 = ap21.apdict(overflow=overflow)
            plt.close(fig1)
            plt.close(fig2)
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

fig, axs = plt.subplots(2, 3, num='figctopt', clear=True, figsize=[9, 3.5], sharey='row', gridspec_kw=dict(height_ratios=[2, 1]))

paramgetter = afterpulse_tile21.Plotter.paramgetter

model_params = [
    ('mu_borel', 'Borel', ''    , '#000'),
    ('p_geom'  , 'Geom.', 'geom', '#f55'),
]

config_laser = sum([
    [
        (f'{name} OF', paramgetter(f'mainfitof{infix}'  , param), dict(color=color, marker='o')),
        ('No OF'     , paramgetter(f'mainfit{infix}'    , param), dict(color=color, marker='o', markerfacecolor='#fff')),
        ('OF, FZ'    , paramgetter(f'mainfitof{infix}fz', param), dict(color=color, marker='s')),
        ('No OF, FZ' , paramgetter(f'mainfit{infix}fz'  , param), dict(color=color, marker='s', markerfacecolor='#fff')),
    ]
    for param, name, infix, color in model_params
], [])

config_ap = sum([
    [
        (f'{name} OF', paramgetter(f'apfitof{infix}', param), dict(color=color, marker='o')),
        ('No OF'     , paramgetter(f'apfit{infix}'  , param), dict(color=color, marker='o', markerfacecolor='#fff')),
    ]
    for param, name, infix, color in model_params
], [])

for iconfig, config in enumerate([config_laser, config_ap]):
    for ivov, (vov, results) in enumerate(vovdict.items()):
        for i, (_, getter, style) in enumerate(config):
            param = getter(results)
            n = len(config) - 1
            offset = 0.075 * (i - n/2)
            axs[iconfig, ivov].errorbar(gvar.mean(param), i, xerr=gvar.sdev(param), capsize=4, **style)

    ax = axs[iconfig, 0]
    ax.set_yticks(list(range(len(config))))
    ax.set_yticklabels([label for label, _, _ in config])
    ax.set_ylim(len(config), -1)

for ivov, vov in enumerate(vovdict):
    textbox.textbox(axs[0, ivov], f'Laser {vov} VoV', fontsize='medium', loc='upper right')
    textbox.textbox(axs[1, ivov], f'AP {vov} VoV', fontsize='medium', loc='lower right')

for ax in axs.flat:
    if ax.is_last_row():
        ax.set_xlabel('Parameter ($\\mu_B$ or $p$)')
    ax.minorticks_on()
    ax.set_yticks([], minor=True)
    ax.grid(which='major', axis='x', linestyle='--')
    ax.grid(which='minor', axis='x', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
