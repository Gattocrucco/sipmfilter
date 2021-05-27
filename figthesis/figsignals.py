import numpy as np
from matplotlib import pyplot as plt

import readwav
import single_filter_analysis
import integrate
import figlatex

nphotons = [1, 3, 5]
length = 2000
leftmargin = 100
rep = 1
filename = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'

###########################

data = readwav.readwav(filename, mmap=False)
mask = ~readwav.spurious_signals(data)

# Run a moving average filter to find and separate the signals by
# number of photoelectrons.
trigger, baseline, value = integrate.filter(data, bslen=8000, length_ma=1470, delta_ma=1530)
corr_value = baseline - value[:, 0]
snr, center, width = single_filter_analysis.single_filter_analysis(corr_value[mask], return_full=True)
assert snr > 15
assert len(center) > 2

# Select the data corresponding to a different number of photoelectrons.
nphotons = np.sort(nphotons)
datanph = []
for nph in nphotons:
    lower = (center[nph - 1] + center[nph]) / 2
    upper = (center[nph] + center[nph + 1]) / 2
    selection = (lower < corr_value) & (corr_value < upper) & mask
    indices = np.flatnonzero(selection)[:rep]
    indices0 = indices[:, None]
    indices2 = trigger[indices, None] + np.arange(-leftmargin, length)
    datanph.append(data[indices0, 0, indices2])

fig, ax = plt.subplots(num='figsignals', clear=True, figsize=[6.4, 3.5])

# linestyles = ['-', '--', '-.', ':']
for i in range(len(datanph)):
    nph = nphotons[i]
    kw = dict(
        # color=f'C{i}',
        color=[i / len(datanph)] * 3,
        label=f'{nph} pe',
        zorder=10 - i,
        # linestyle=linestyles[i],
    )
    for j in range(len(datanph[i])):
        kw['alpha'] = 1 - j / len(datanph[i])
        ax.plot(datanph[i][j], **kw)
        kw.pop('label', None)

ax.legend(loc='best')
ax.set_xlabel('Time [ns]')
ax.set_ylabel('ADC scale [10 bit]')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
