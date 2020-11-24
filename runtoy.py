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

tau = np.array([4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384])

# snr hardcoded here. To compute it again, do:
# t = toy.Toy(..., tau=np.array([128]), snr=10)
# snr128 = t.snr[0]
snr128 = np.array([1.7658007 , 2.06268908, 2.35957745, 2.65646583, 2.9533542 ,
                   3.25024257, 3.54713095, 3.84401932, 4.1409077 , 4.43779607])

snr = np.empty(tau.shape + snr128.shape)
snr[:] = snr128
t = toy.Toy(data, ~ignore, tau, snr, bslen=1024, bsoffset=32)
out, oute = t.run(1000, 'runtoy.npy', pbar=10, seed=0)
