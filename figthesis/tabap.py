from matplotlib import pyplot as plt
import numpy as np

import afterpulse_tile21
import textmatrix
import uncertainties

table = []

def fitbkg(fit):
    x = fit.data[0]
    bins = x['bins']
    t = bins[-1] - bins[0]
    c = fit.palt['const']
    ct = c * t
    count = x['norm']
    return count * ct / (1 + ct)

def uformat(x):
    ux = uncertainties.ufloat(x.mean, x.sdev)
    return f'{ux:.2u}'.replace('+/-', ' \\pm ')

def pformat(p, limit=1e-6):
    if p < limit:
        return f'{{<{limit:.1g}}}'
    else:
        return f'{p:.2g}'

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):
    
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    (fit1, fit2), fig1, fig2 = ap21.apfittau()
    plt.close(fig1)
    plt.close(fig2)
    
    # table columns:
    #     Delay cut                       Fit parameters                 Fit quality
    # vov lcut rcut events count time bkg const tau1 tau2 p1 factor prob chi2 dof pv
    row1 = [
        vov,
        ap21.params['dcut'],
        ap21.params['dcutr'],
        ap21.apnevents,
        int(ap21.apcount.mean),
        f'{ap21.aptime:.2g}',
        ap21.apbkg,
    ]
    row2a = [
        fitbkg(fit1),
        fit1.palt['tau'],
        '{n.d.}',
        '{n.d.}',
        ap21.apfactor,
        ap21.approb * 100,
        f'{fit1.chi2:.2g}',
        fit1.dof,
        pformat(fit1.Q),
    ]
    row2b = [
        fitbkg(fit2),
        fit2.palt['tau'][0],
        fit2.palt['tau'][1] * 1e-3,
        fit2.palt['w0'] * 100,
        ap21.apfactor2,
        ap21.approb2 * 100,
        f'{fit2.chi2:.2g}',
        fit2.dof,
        pformat(fit2.Q),
    ]
    table += [
        row1 + row2a,
        row1 + row2b,
    ]

for row in table:
    for i, x in enumerate(row):
        if hasattr(x, 'mean'):
            row[i] = uformat(x)

matrix = textmatrix.TextMatrix(table)
print(matrix.latex())
