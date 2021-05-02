import numpy as np
from matplotlib import pyplot as plt

import readwav
import single_filter_analysis
import integrate
import figlatex

filename = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
length = 128

###########################

data = readwav.readwav(filename, mmap=False)
mask = ~readwav.spurious_signals(data)
trigger, baseline, value = integrate.filter(data, bslen=8000, length_ma=length, delta_ma=length)
corr_value = baseline - value[:, 0]

fig = plt.figure(num='figfingerplot', clear=True)
single_filter_analysis.single_filter_analysis(corr_value[mask], fig)

fig.tight_layout()
fig.show()

figlatex.save(fig)
