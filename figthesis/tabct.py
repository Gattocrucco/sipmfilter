import os

from matplotlib import pyplot as plt
import numpy as np
import uncertainties
import gvar

import afterpulse_tile21
import textmatrix

cache = 'figthesis/tabct.gvar'

################

gvar.switch_gvar()

if not os.path.exists(cache):
    
    vovdict = {}
    
    for vov in afterpulse_tile21.AfterPulseTile21.defaultparams:
        ap21 = afterpulse_tile21.AfterPulseTile21(vov)
        
        _, fig1, fig2 = ap21.ptdict()
        plt.close(fig1)
        plt.close(fig2)
        
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

def uformat(x):
    ux = uncertainties.ufloat(x.mean, x.sdev)
    return f'{ux}'.replace('+/-', ' \\pm ')

def pformat(p, limit=1e-6):
    if p < limit:
        return f'{{<{limit:.1g}}}'.replace('e-0', 'e-')
    else:
        return f'{p:#.2f}'

def getp(fit, param):
    if param in fit.palt.extension_keys():
        mu = fit.palt[param]
        if fit.Q < 0.01:
            factor = np.sqrt(fit.chi2 / fit.dof)
            mu = afterpulse_tile21.scalesdev(mu, factor)
        return mu
    else:
        return ''

def probpe(fit):
    if 'mu_borel' in fit.palt.extension_keys():
        mu = getp(fit, 'mu_borel')
        p = 1 - np.exp(-mu)
    else:
        mu = getp(fit, 'p_geom')
        p = mu
    return 100 * p, 1 / (1 - mu) - 1

# table columns:
# OV kind N OF FZ model mub pg mup prob pe chi2 dof pvalue
tables = []

for vov, results in vovdict.items():
    table = []
    for data, dt in [('Random', 'pt'), ('Laser', 'main'), ('AP', 'ap')]:
        for model, md in [('Borel', ''), ('Geom.', 'geom')]:
            for fixzero, fz in [('No', ''), ('Yes', 'fz')]:
                for overflow, of in [('Yes', 'of'), ('No', '')]:
        
                    fitkey = dt + 'fit' + of + md + fz
                    if fitkey not in results:
                        continue
                    
                    fit = results[fitkey]
                    row = [
                        vov,
                        data,
                        fit.x['norm'],
                        'Yes' if dt == 'pt' else overflow,
                        fixzero if dt == 'main' else '',
                        model,
                        getp(fit, 'mu_borel'),
                        getp(fit, 'p_geom'),
                        getp(fit, 'mu_poisson'),
                        *probpe(fit),
                        f'{fit.chi2:.0f}',
                        fit.dof,
                        pformat(fit.Q),
                    ]
                    table.append(row)
    for row in table:
        for i, x in enumerate(row):
            if hasattr(x, 'sdev'):
                row[i] = uformat(x)
    tables.append(table)

for table in tables:
    matrix = textmatrix.TextMatrix(table)
    print(matrix.latex() + ' \\\\')
    print('\\midrule')
