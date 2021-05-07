from matplotlib import pyplot as plt
import gvar
import uncertainties

import figlatex
import afterpulse_tile21
import textmatrix

def ulatex(x):
    x = uncertainties.ufloat(gvar.mean(x), gvar.sdev(x))
    return str(x).replace('+/-', ' \\pm ')

fig, ax = plt.subplots(num='figptrate', clear=True, figsize=[2.4, 3])

table = []

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):

    ap21 = afterpulse_tile21.AfterPulseTile21(vov)
    
    # dblcount = ap21.sim.getexpr(f"""count_nonzero(
    #     (length == {ap21.params['ptlength']})
    #     & (ptApos >= {ap21.params['lmargin']})
    #     & (ptApos < trigger - {ap21.params['rmargin']})
    #     & (ptAamplh > {ap21.ptcut})
    #     & (ptBpos >= {ap21.params['lmargin']})
    #     & (ptBpos < trigger - {ap21.params['rmargin']})
    #     & (ptBamplh > {ap21.ptcut})
    # )""")
    
    # table columns:
    # vov, nevents, time: per event, total; pre-trigger: count, rate
    table.append([
        vov,
        ap21.ptnevents,
        f'{ap21.pttime / ap21.ptnevents * 1e6:.3g}',
        f'{ap21.pttime:.3g}',
        int(gvar.mean(ap21.ptcount)),
        ulatex(ap21.ptrate),
    ])
    
    ax.errorbar(vov, gvar.mean(ap21.ptrate), gvar.sdev(ap21.ptrate), color='black', capsize=4, marker='.')

ax.set_xlabel('Overvoltage [V]')
ax.set_ylabel('Pre-trigger rate [cps]')

ax.minorticks_on()
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle=':')
ax.set_ylim(0, ax.get_ylim()[1])

fig.tight_layout()
fig.show()

table = textmatrix.TextMatrix(table)
print(table.latex())

figlatex.save(fig)
