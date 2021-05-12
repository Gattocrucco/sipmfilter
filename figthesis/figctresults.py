from matplotlib import pyplot as plt

import figlatex
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

fig, axs = plt.subplots(1, 3, num='figctresults', clear=True, figsize=[9, 3])
plotter = afterpulse_tile21.Plotter(vovdict)
plotter.plotdictparam(axs[0])
plotter.plotdictprob(axs[1])
plotter.plotdictpe(axs[2])

ax = axs[0]
ax.set_title('DiCT parameter')
ax.set_ylabel('Branching parameter ($\\mu_B$ or $p$)')

ax = axs[1]
ax.set_title('DiCT probability')
ax.set_ylabel('Prob. of > 1 pe [%]')

ax = axs[2]
ax.set_title('DiCT pe')
ax.set_ylabel('Average excess pe')

for ax in axs:
    if not ax.is_first_col():
        ax.legend().set_visible(False)
    ax.set_xlabel('Overvoltage [V]')

fig.tight_layout()
fig.show()

figlatex.save(fig)
