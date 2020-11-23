import numpy as np
from matplotlib import pyplot as plt

import toy
import readwav

generator = np.random.default_rng(202011111750)

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False)
ignore = readwav.spurious_signals(data)
print(f'ignoring {np.sum(ignore)} events with signals in baseline zone')

def plot_mf_template():
    template = toy.Template()
    template.make(data, 4096, ~ignore)

    fig = plt.figure('runtoy-mf-template')
    fig.clf()

    ax = fig.subplots(1, 1)

    template_offset = [
        template.matched_filter_template(length, norm=False)
        for length in [4, 8, 16, 32, 64]
    ]
    for y, offset in reversed(template_offset):
        ax.plot(np.arange(len(y)) + offset, y, label=str(len(y)))

    ax.set_title('Matched filter template for different lengths')
    ax.set_xlabel('Sample number @ 125 MSa/s')
    ax.set_ylabel('ADC scale')
    ax.legend(loc='best')
    ax.grid()

    fig.tight_layout()
    fig.show()

def plot_signals():
    template = toy.Template()
    template_length = 512 # @ 125 MSa/s
    template.make(data, template_length * 8, ~ignore)

    event_length = 2 ** 11 # @ 125 MSa/s
    signal_loc = generator.integers(event_length - template_length, size=4)
    simulated_signal = template.generate(event_length, signal_loc, generator)
    simulated_noise = toy.Noise().generate(len(signal_loc), event_length, generator)

    fig = plt.figure('runtoy-plot-signals')
    fig.clf()

    ax = fig.subplots(1, 1)

    for i in range(len(signal_loc)):
        ax.plot(simulated_signal[i] + 5 * simulated_noise[i])

    ax.set_title('Simulated signals')
    ax.set_xlabel('Sample number @ 125 MSa/s')
    ax.set_ylabel('ADC scale')
    ax.grid()

    fig.tight_layout()
    fig.show()

def plot_filters():
    template = toy.Template()
    template_length = 512 # @ 125 MSa/s
    template.make(data, template_length * 8, ~ignore)

    event_length = 2 ** 11 # @ 125 MSa/s
    signal_loc = generator.integers(event_length - template_length, size=1)
    simulated_signal = template.generate(event_length, signal_loc, generator)
    simulated_noise = toy.Noise().generate(len(signal_loc), event_length, generator)
    simulation = simulated_signal + 5 * simulated_noise

    filt = toy.Filter(simulation, template.baseline)
    length = 64
    filt_ma = filt.moving_average(length)
    filt_exp = filt.exponential_moving_average(length)
    mf_templ, mf_offset = template.matched_filter_template(length)
    filt_mf = filt.matched(mf_templ)

    fig = plt.figure('runtoy-filters')
    fig.clf()

    ax = fig.subplots(1, 1)

    ax.plot(simulation[0], label='signal')
    ax.plot(filt_ma[0], label='moving average')
    ax.plot(filt_exp[0], label='exponential moving average')
    ax.plot(np.arange(event_length) + mf_offset, filt_mf[0], label='matched filter')

    ax.legend(loc='best')
    ax.set_title('Simulated signal and filtering')
    ax.set_xlabel('Sample number @ 125 MSa/s')
    ax.set_ylabel('ADC scale')
    ax.grid()

    fig.tight_layout()
    fig.show()

def plot_localization():
    snr = 5
    tau = 64

    template = toy.Template()
    template_length = tau + 32 # @ 125 MSa/s
    template.make(data, template_length * 8, ~ignore)

    event_length = 2 ** 11 # @ 125 MSa/s
    signal_loc = generator.integers(event_length - template_length, size=1)
    simulated_signal = template.generate(event_length, signal_loc, generator)
    simulated_noise = toy.Noise().generate(len(signal_loc), event_length, generator)
    noise_sigma = template.maximum / snr

    mf_templ, mf_offset = template.matched_filter_template(tau)
    filt_noise = toy.Filter(simulated_noise)
    noise = filt_noise.all(mf_templ)
    filt_signal = toy.Filter(simulated_signal, template.baseline)
    signal = filt_signal.all(mf_templ)
    sim = signal + noise_sigma * noise

    temp_loc = np.argmin(sim, axis=-1)
    temp_loc = np.array(temp_loc, float)
    temp_loc[3] += mf_offset

    fig = plt.figure('runtoy-localization')
    fig.clf()

    ax = fig.subplots(1, 1)

    line, = ax.plot(sim[0, 0], label='simulation')
    ax.axvline(temp_loc[0, 0], color=line.get_color())
    line, = ax.plot(sim[1, 0], label='moving average')
    ax.axvline(temp_loc[1, 0], color=line.get_color())
    line, = ax.plot(sim[2, 0], label='exponential moving average')
    ax.axvline(temp_loc[2, 0], color=line.get_color())
    line, = ax.plot(np.arange(event_length) + mf_offset, sim[3, 0], label='matched filter')
    ax.axvline(temp_loc[3, 0], color=line.get_color())

    ax.legend(loc='best')
    ax.set_title('Simulated signal and filtering')
    ax.set_xlabel('Sample number @ 125 MSa/s')
    ax.set_ylabel('ADC scale')
    ax.grid()

    fig.tight_layout()
    fig.show()

def plot(ifilter, itau, isnr, ievent):
    mf_templ, mf_offset = template.matched_filter_template(tau[itau])
    noise = filt_noise.all(mf_templ)[ifilter, ievent]
    signal = filt_signal.all(mf_templ)[ifilter, ievent]
    sigma = template.maximum / snr[itau, isnr]
    sim = signal + sigma * noise
    assert sim.shape == (event_length,)
    
    minimum = np.argmin(sim)
    minval = np.min(sim)
    
    unfilt = simulated_signal[ievent] + sigma * simulated_noise[ievent]
    
    fig = plt.figure('runtoy-plot')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    ax.plot(unfilt, label='signal')
    ax.plot(sim, label=f'filter ({ifilter})')
    ax.axvline(signal_loc[ievent], label='signal template start', color='black')
    ax.axvline(loc[ifilter, itau, isnr, ievent], label='localization (uncalib.)', color='red', linestyle='--')
    ax.axhline(baseline[ifilter, itau, isnr, ievent], label='baseline', color='black', linestyle='--')
    ax.grid()
    ax.legend(loc='best', fontsize='small')
    
    fig.show()

tau = np.array([4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256])

# snr hardcoded here for the tau array above. to compute it again, do:
# t = toy.Toy(..., snr=10)
# snr = t.snr
snr = np.array([[3.64666879, 3.73457182, 3.82247485, 3.91037788, 3.99828091,
                 4.08618394, 4.17408698, 4.26199001, 4.34989304, 4.43779607],
                [2.96251777, 3.12643758, 3.29035739, 3.4542772 , 3.61819701,
                 3.78211682, 3.94603664, 4.10995645, 4.27387626, 4.43779607],
                [2.3306752 , 2.56479974, 2.79892428, 3.03304882, 3.26717336,
                 3.5012979 , 3.73542245, 3.96954699, 4.20367153, 4.43779607],
                [2.05047304, 2.31573115, 2.58098927, 2.84624738, 3.1115055 ,
                 3.37676361, 3.64202173, 3.90727984, 4.17253796, 4.43779607],
                [1.89391265, 2.17656637, 2.45922008, 2.74187379, 3.02452751,
                 3.30718122, 3.58983493, 3.87248864, 4.15514236, 4.43779607],
                [1.80128303, 2.09422892, 2.38717481, 2.68012071, 2.9730666 ,
                 3.2660125 , 3.55895839, 3.85190428, 4.14485018, 4.43779607],
                [1.74531611, 2.04448055, 2.34364499, 2.64280943, 2.94197387,
                 3.24113831, 3.54030275, 3.83946719, 4.13863163, 4.43779607],
                [1.69182299, 1.99693111, 2.30203923, 2.60714735, 2.91225547,
                 3.21736359, 3.52247171, 3.82757983, 4.13268795, 4.43779607],
                [1.70364215, 2.00743703, 2.31123191, 2.61502679, 2.91882167,
                 3.22261655, 3.52641143, 3.83020631, 4.13400119, 4.43779607],
                [1.7658007 , 2.06268908, 2.35957745, 2.65646583, 2.9533542 ,
                 3.25024257, 3.54713095, 3.84401932, 4.1409077 , 4.43779607],
                [1.9738837 , 2.24765174, 2.52141978, 2.79518782, 3.06895586,
                 3.3427239 , 3.61649195, 3.89025999, 4.16402803, 4.43779607],
                [2.23307196, 2.47804131, 2.72301065, 2.96798   , 3.21294934,
                 3.45791869, 3.70288803, 3.94785738, 4.19282672, 4.43779607]])

t = toy.Toy(data, ~ignore, tau, snr, bslen=1024, bsoffset=32)
out, oute = t.run(1000, 'runtoy.npy', pbar=10, seed=0)

def plot_loc(itau, isnr):
    fig = plt.figure('runtoy-loc')
    fig.clf()

    axs = fig.subplots(2, 2).reshape(-1)
    
    names = [
        f'No filter. tau = {tau[itau]}, SNR = {snr[itau, isnr]:.2f}',
        'Moving average',
        'Exponential moving average',
        'Matched filter'
    ]
    
    for i, (ax, name) in enumerate(zip(axs, names)):
        data = out['loc'][:, i, itau, isnr]
        ax.hist(data, bins='auto', histtype='step')
        if ax.is_last_row():
            ax.set_xlabel('Signal localization [8 ns]')
        if ax.is_first_col():
            ax.set_ylabel('Bin count')
        ax.grid()
        ax.set_title(name)

    fig.show()

def plot_locall():
    loctrue = out['loctrue'][:, None, None, None]
    loc = out['loc']
    quantiles = np.quantile(loc - loctrue, [0.5 - 0.68/2, 0.5 + 0.68/2], axis=0)
    width = (quantiles[1] - quantiles[0]) / 2
    assert width.shape == (4,) + snr.shape
    
    fig = plt.figure('runtoy-locall')
    fig.clf()

    axs = fig.subplots(2, 2, sharex=True, sharey=True).reshape(-1)
    
    names = [
        'No filter',
        'Moving average',
        'Exponential moving average',
        'Matched filter'
    ]
    
    for ifilter, (ax, name) in enumerate(zip(axs, names)):
        for itau in range(len(tau)):
            alpha = (itau + 1) / len(tau)
            label = f'tau = {tau[itau]}'
            ax.plot(snr[itau], width[ifilter, itau], color='black', alpha=alpha, label=label)
        if ax.is_last_row():
            ax.set_xlabel('SNR (avg max sig ampl over noise rms)')
        if ax.is_first_col():
            ax.set_ylabel('"1$\\sigma$" interquantile range')
        ax.grid()
        ax.set_title(name)
        ax.legend(loc='best', fontsize='small')
    
    axs[0].set_yscale('symlog', linthreshy=1, linscaley=0.5)
    
    fig.tight_layout()
    fig.show()
