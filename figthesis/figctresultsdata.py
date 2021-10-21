from matplotlib import pyplot as plt
import numpy as np
import gvar

import afterpulse_tile21

vovdict = {}
for vov in afterpulse_tile21.AfterPulseTile21.defaultparams:
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)
    for meth in ['apdict', 'maindict', 'ptdict']:
        if meth == 'ptdict':
            _, fig1, fig2 = getattr(ap21, meth)()
        else:
            for overflow in [True, False]:
                _, fig1, fig2 = getattr(ap21, meth)(overflow=overflow)
        plt.close(fig1)
        plt.close(fig2)
    vovdict[vov] = ap21.results

plotter = afterpulse_tile21.Plotter(vovdict)

transfsborel = [
    lambda mu: mu,
    lambda mu: 100 * (1 - np.exp(-mu)),
    lambda mu: 1 / (1 - mu) - 1,
]
transfsgeom = [
    lambda p: p,
    lambda p: 100 * p,
    lambda p: 1 / (1 - p) - 1,
]
titles = [
    'parameter',
    'probability',
    'pe'
]

muborel = [
    ('Random', plotter.paramgetter( 'ptfit'    , 'mu_borel')),
    ('Laser' , plotter.paramgetter(f'mainfitof', 'mu_borel')),
    ('AP'    , plotter.paramgetter(f'apfitof'  , 'mu_borel')),
]
mugeom = [
    ('Random', plotter.paramgetter( 'ptfitgeom'    , 'p_geom')),
    ('Laser' , plotter.paramgetter(f'mainfitofgeom', 'p_geom')),
    ('AP'    , plotter.paramgetter(f'apfitofgeom'  , 'p_geom')),
]

print('# figure 7.27')
for transfborel, transfgeom, title in zip(transfsborel, transfsgeom, titles):
    print(f'\n# panel "{title}"')
    for mus, color, f, model in zip([muborel, mugeom], ['black', 'red'], [transfborel, transfgeom], ['borel', 'geom']):
        print(f'\n# model {model} (color: {color})')
        for (label, getter), marker in zip(mus, ['triangle', 'none', 'dot']):
            print(f'\n# {label} (marker: {marker})')
            print('# VOV\tparam\terror')
            param = plotter.listdict(getter)
            param = f(param)
            for x, y in zip(plotter.vov, param):
                print(f'{x}\t{gvar.mean(y):.4f}\t{gvar.sdev(y):.4f}')
