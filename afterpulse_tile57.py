import os

import numpy as np
from scipy import stats

import afterpulse
import readwav
import toy

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'

data = readwav.readwav(prefix + '.wav')
template = toy.Template.load(prefix + '-template.npz')

savedir = 'afterpulse_tile57'
os.makedirs(savedir, exist_ok=True)

simfile =f'{savedir}/sim.npz'
if not os.path.exists(simfile):
    sim = afterpulse.AfterPulse(data, template, batch=100, pbar=True)
    sim.save(simfile)
sim = afterpulse.AfterPulse.load(simfile)

cond = 'good&(mainpos>=0)&(length==512)&(npe>0)&(npe<8)'
fig = sim.hist('mainheight/npe', cond)
fig.savefig(f'{savedir}/fig1.png')
m = sim.getexpr('median(mainheight/npe)', cond)
print(f'fig1 median = {m}')

cond = 'good&(mainpos>=0)&(length==512)&(npe==1)'
fig = sim.scatter('minorpos-mainpos', 'minorheight', cond)
fig.savefig(f'{savedir}/fig2.png')
nscatter = sim.getexpr(f'count_nonzero({cond})')
dot1pecond = f'{cond}&(minorheight>39)&(minorheight<50)'
dot1pe = sim.eventswhere(dot1pecond)
count = len(dot1pe)
dothigh = sim.eventswhere(f'{cond}&(minorheight>70)')
print(f'fig2 total = {nscatter}')
print(f'fig2 dots at 1 pe = {count}')
print(f'fig2 high dots = {len(dothigh)}')

fig = sim.plotevent(data, dot1pe[0], 8)
fig.savefig(f'{savedir}/fig3.png')

fig = sim.hist('minorpos-mainpos', dot1pecond)
fig.savefig(f'{savedir}/fig4.png')
dist = sim.getexpr('minorpos-mainpos', dot1pecond)
m = np.mean(dist)
s = np.std(dist)
l = np.min(dist)
tau = (m - l)
dtau = s / np.sqrt(len(dist))
print(f'fig4 expon tau = {tau} +/- {dtau}')

cond = 'good&(mainpos>=0)&(length==2048)&(npe>0)&(npe<8)'
fig = sim.hist('mainheight/npe', cond)
fig.savefig(f'{savedir}/fig5.png')
m = sim.getexpr('median(mainheight/npe)', cond)
print(f'fig5 median = {m}')

cond = '(length==2048)&(ptpos>=0)'
fig = sim.hist('ptheight', '(length==2048)&(ptpos>=0)', 'log')
ax, = fig.get_axes()
ax.set_xlim(0, 100)
fig.savefig(f'{savedir}/fig6.png')
cond1pe = f'{cond}&(ptheight>20)&(ptheight<40)'
evt1pe = sim.eventswhere(cond1pe)
n = len(evt1pe)
trigger = sim.getexpr('median(trigger)')
time = len(sim.output) * trigger * 1e-9
rate = n / time
drate = np.sqrt(n) / time
print(f'fig6 1pe (between 20 and 40) counts = {n}')
print(f'fig6 rate = {rate} +/- {drate}')

fig = sim.plotevent(data, evt1pe[0], zoom='all')
fig.savefig(f'{savedir}/fig7.png')

dcount = np.sqrt(count)
print(f'count = {count} +/- {dcount}')

p = stats.expon.cdf(l, scale=500)
corr = p / (1 - p)
print(f'correction factor for exponential truncation = {corr}')

psamp = data.shape[-1] - trigger
ptime = nscatter * psamp * 1e-9
bcount = ptime * rate
dbcount = ptime * drate
print(f'expected random counts = {bcount} +/- {dbcount}')

fcount = count * corr - bcount
dfcount = np.hypot(dcount * corr, dbcount)
print(f'corrected count = {fcount} +/- {dfcount}')

frac = fcount / nscatter
dfrac = dfcount / nscatter
print(f'total npe==1 = {nscatter}')
print(f'fraction = {frac} +/- {dfrac}')
