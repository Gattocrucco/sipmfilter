from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21

vovdict = {}
for vov in afterpulse_tile21.AfterPulseTile21.defaultparams:
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)
    _, fig1, fig2 = ap21.apfittau()
    plt.close(fig1)
    plt.close(fig2)
    ap21.approb
    ap21.approb2
    vovdict[vov] = ap21.results

fig, axs = plt.subplots(1, 3, num='figapresults', clear=True, figsize=[9, 3])
plotter = afterpulse_tile21.Plotter(vovdict)
plotter.plotapprob(axs[0])
plotter.plotaptau(axs[1])
plotter.plotapweight(axs[2])

ax = axs[0]
ax.set_title('AP probability')
ax.set_ylabel('Prob. of $\\geq$1 ap after 1 pe signal [%]')
ax.legend(fontsize='small')

ax = axs[1]
ax.set_title('AP decay')
ax.set_ylabel('Exponential decay constant [ns]')
ax.legend(fontsize='small')

ax = axs[2]
ax.set_title('AP mixture')
ax.set_ylabel('Weight of short component')
ax.set_ylim(0, 1)

for ax in axs:
    ax.set_xlabel('Overvoltage [V]')

fig.tight_layout()
fig.show()

figlatex.save(fig)
