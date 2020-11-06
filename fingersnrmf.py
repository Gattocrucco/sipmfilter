import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, optimize
import tqdm

import readwav
import integrate
from single_filter_analysis import single_filter_analysis

# Load wav file.
filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False, maxevents=None)
ignore = readwav.spurious_signals(data)
print(f'ignoring {np.sum(ignore)} events with signals in baseline zone')

def make_template(data, ignore=None, length=2000, fig=None):
    """
    Make a template waveform for the matched filter.
    
    Parameters
    ----------
    data : array (nevents, 2, 15001)
        As returned by readwav.readwav().
    ignore : bool array (nevents,), optional
        Flag events to be ignored.
    length : int
        Number of samples of the waveform.
    fig : matplotlib figure, optional
        If given, plot the waveform.
    
    Return
    ------
    waveform : array (length,)
        The waveform. It is normalized to unit sum.
    """
    
    if ignore is None:
        ignore = np.zeros(len(data), bool)

    # Run a moving average filter to find and separate the signals by number of
    # photoelectrons.
    trigger, baseline, value = integrate.filter(data, bslen=8000, length_ma=1470, delta_ma=1530)
    corr_value = baseline - value[:, 0]
    snr, center, width = single_filter_analysis(corr_value[~ignore], return_full=True)
    assert snr > 15
    assert len(center) > 2
    
    # Select the data corresponding to 1 photoelectron and subtract the
    # baseline.
    lower = (center[0] + center[1]) / 2
    upper = (center[1] + center[2]) / 2
    selection = (lower < corr_value) & (corr_value < upper) & ~ignore
    t = int(np.median(trigger))
    data1pe = data[selection, 0, t:t + length] - baseline[selection].reshape(-1, 1)
    
    # Compute the waveform as the median of the signals.
    waveform, bottom, top = np.quantile(data1pe, [0.5, 0.25, 0.75], axis=0)
    # waveform = np.mean(data1pe, axis=0)
    # std = np.std(data1pe, axis=0)
    # bottom = waveform - std
    # top = waveform + std
    
    if fig is not None:
        ax = fig.subplots(1, 1)

        ax.fill_between(np.arange(length), bottom, top, facecolor='lightgray', label='25%-75% quantiles')
        ax.plot(waveform, 'k-', label='median')
        
        ax.grid()
        ax.legend(loc='best')
        ax.set_xlabel('Sample number')
        ax.set_ylabel('ADC value')
        ax.set_title('Template for matched filter (unnormalized)')
    
    return waveform / np.sum(waveform)

fig = plt.figure('fingersnrmf')
fig.clf()

waveform = make_template(data, ignore=ignore, fig=fig)

fig.tight_layout()
fig.show()
    
def snrseries(bslen=6900):
    """
    Compute SNR as a function of tau and delta for some values of tau and delta
    hardcoded in the script.
    
    Parameters
    ----------
    bslen : int
        The number of samples used for the baseline.
    
    Returns
    -------
    tau : array (ntau,)
        Values of the filter scale parameter.
    delta_ma : array (ntau, ndelta)
        Values of the offset for the moving average for each tau.
    delta_exp : array (ntau, ndelta)
        Values of the offset for the moving exponential average for each tau.
    snr : array (2, ntau, ndelta)
        The SNR for (moving average, exponential moving average), and for each
        length parameter (tau) and offset from trigger (delta).
    """
    print('computing filters...')
    start, baseline, vma, vexp = integrate.filter(data, bslen, delta, tau, delta_exp, tau)

    snr = np.empty((2, len(tau)))

    print('analysing filter output...')
    for i in tqdm.tqdm(range(snr.shape[1])):
        for j, value in enumerate([vma, vexp]):
            value = value[:, i]
            corr_value = (baseline - value)[~ignore]
            snr[j, i] = single_filter_analysis(corr_value)

    snr = snr.reshape(2, -1, ndelta)
    ltau = tau.reshape(-1, ndelta)
    ldelta = delta.reshape(ltau.shape)
    ldelta_exp = delta_exp.reshape(ltau.shape)
    
    output = (ltau[:, 0], ldelta, ldelta_exp, snr)
    fig = snrplot(*output)
    return output

def snrplot(tau, delta_ma, delta_exp, snr):
    """
    Plot SNR as a function of tau and delta for some values of tau and delta
    hardcoded in the script. Called by snrseries().
    
    Parameters
    ----------
    The output from snrseries().
    
    Returns
    -------
    fig : matplotlib figure
        The figure with the plots.
    """

    fig = plt.figure('fingersnr-snrplot', figsize=[6.4, 7.1])
    fig.clf()

    axs = fig.subplots(2, 1, sharey=True, sharex=True)

    axs[0].set_title('Moving average')
    axs[1].set_title('Exponential moving average')
    axs[1].set_xlabel('Offset from trigger [ns]')

    for i, ax in enumerate(axs):
        for j in range(len(tau)):
            alpha = 1 - (j / len(tau))
            label = f'tau = {tau[j]} ns'
            d = delta_ma if i == 0 else delta_exp
            ax.plot(d[j], snr[i, j], color='black', alpha=alpha, label=label)
        ax.set_ylabel('SNR')
        ax.legend(loc='best', fontsize='small')
        ax.grid()

    fig.tight_layout()
    fig.show()
    
    return fig

def fingerplot(tau, delta, kind='ma', bslen=6900):
    start, baseline, vma, vexp = integrate.filter(data, bslen, delta, tau, delta, tau)
    value = dict(ma=vma, exp=vexp)[kind][:, 0]
    corr_value = (baseline - value)[~ignore]
    fig1 = plt.figure('fingersnr-fingerplot-1', figsize=[7.27, 5.73])
    fig2 = plt.figure('fingersnr-fingerplot-2', figsize=[6.4, 4.8])
    fig1.clf()
    fig2.clf()
    snr = single_filter_analysis(corr_value, fig1, fig2)
    print(f'snr = {snr:.1f}')
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.show()
    fig2.show()
    
# Optimization does not work very well for the exponential moving average
# because yes, while it seems to be ok for the moving average.
#
# def fun(x, useexp):
#     delta, tau = x
#     delta = np.array([delta], int)
#     tau = np.array([tau], int)
#     start, baseline, vma, vexp = integrate.filter(data, delta, tau, delta, tau)
#     value = vexp if useexp else vma
#     corr_value = (baseline - value[:, 0])[~ignore]
#     snr = single_filter_analysis(corr_value)
#     print(snr)
#     return -snr
#
# print('searching optimal parameters for moving average...')
# options = dict(maxfev=100, disp=True, return_all=True, xatol=1, fatol=0.001)
# resultma = optimize.minimize(fun, x0=[1470, 1530], args=(False,), options=options, method='Nelder-Mead')
#
# print('searching optimal parameters for exponential average...')
# options = dict(maxfev=100, disp=True, return_all=True, xatol=1, fatol=0.001)
# resultexp = optimize.minimize(fun, x0=[1500, 800], args=(True,), options=options, method='Nelder-Mead')

def snrmax(bslen=6900):
    """
    Find the maximum SNR varying delta for each tau. Tau values to consider are
    hardcoded in the script. "Delta" is the offset from the trigger. Also plot
    the results.
    
    Parameters
    ----------
    bslen : int
        The number of samples used for the baseline.
    
    Returns
    -------
    tau : array (N,)
        The tau values tested.
    snrmax : array (2, ntau)
        The maximum SNR for each tau, first dimension is (moving average,
        exponential moving average).
    deltarange : array (2, ntau, 3)
        The triplets [delta_left, delta_max, delta_right] where delta_max is
        the delta that maximizes the SNR and delta_left and _right are points
        where the SNR is -1 relative to the maximum.
    
    """
    def fun(delta, tau, useexp):
        try:
            if useexp:
                start, baseline, value = integrate.filter(data, bslen, delta_exp=delta, tau_exp=tau)
            else:
                start, baseline, value = integrate.filter(data, bslen, delta_ma=delta, length_ma=tau)
        except ZeroDivisionError:
            return 0
        corr_value = (baseline - value[:, 0])[~ignore]
        snr = single_filter_analysis(corr_value)
        return -snr
    
    ltau = tau.reshape(-1, ndelta)

    print('maximizing SNR for each tau...')
    snrmax = np.empty((2, len(ltau)))
    deltarange = np.empty((2, len(ltau), 3))
    # dim0: MOVAVG, EXPAVG
    # dim1: tau
    # dim2: left, max, right
    for i in tqdm.tqdm(range(len(ltau))):
        t = ltau[i, 0]
        for j in range(2):
            useexp = j == 1
            args = (t, useexp)
            bracket = (66 + t * 0.8, 66 + t * 1.2)
            options = dict(xtol=1, maxiter=20)
            result = optimize.minimize_scalar(fun, bracket=bracket, args=args, options=options, method='golden')
            if not result.success:
                print(f'i={i}, j={j}, max: {result}')
            deltamax = result.x
            deltarange[j, i, 1] = deltamax
            snrmax[j, i] = -result.fun
        
            f = lambda *args: fun(*args) - (1 - snrmax[j, i])
            kw = dict(args=args, options=options, method='bisect')
            
            try:
                bracket = (0, deltamax)
                result = optimize.root_scalar(f, bracket=bracket, **kw)
                if not result.converged:
                    print(f'i={i}, j={j}, left: {result}')
                deltarange[j, i, 0] = result.root
            except ValueError:
                deltarange[j, i, 0] = np.nan
            
            try:
                bracket = (deltamax, 3 * deltamax)
                result = optimize.root_scalar(f, bracket=bracket, **kw)
                if not result.converged:
                    print(f'i={i}, j={j}, right: {result}')
                deltarange[j, i, 2] = result.root
            except ValueError:
                deltarange[j, i, 2] = np.nan
    
    output = (ltau[:, 0], snrmax, deltarange)
    snrmaxplot(*output)
    return output

def snrmaxplot(tau, snrmax, deltarange):
    """
    Plot the output from snrmax(). Called by snrmax().
    
    Parameters
    ----------
    The things returned by snrmax().
    
    Returns
    -------
    fig : matplotlib figure
    """
    fig = plt.figure('fingersnr-snrmaxplot', figsize=[6.4, 7.1])
    fig.clf()

    axs = fig.subplots(3, 1, sharex=True)

    axs[0].set_title('Maximum SNR')
    axs[1].set_title('Offset from trigger that maximizes the SNR')
    axs[2].set_title('Interval width -1 SNR relative to maximum')
    axs[0].set_ylabel('SNR')
    axs[1].set_ylabel('Offset from trigger [ns]')
    axs[2].set_ylabel('Offset from trigger [ns]')
    axs[2].set_xlabel('Filter duration parameter [ns]')
    
    for i, label in enumerate(['moving average', 'exponential moving average']):
        x = tau + [-8, 8][i]

        ax = axs[0]
        line, = ax.plot(x, snrmax[i], '.--', label=label)
        color = line.get_color()
        ax.legend(loc='best')
        ax.grid(True)
    
        ax = axs[1]
        dr = deltarange[i].T
        yerr = np.array([
            dr[1] - dr[0],
            dr[2] - dr[1]
        ])
        sel = snrmax[i] > 0
        ax.errorbar(x[sel], dr[1, sel], yerr=yerr[:, sel], fmt='.', color=color, capsize=4)
        ax.grid(True)
        
        ax = axs[2]
        ax.plot(x, dr[2] - dr[0], '.--', color=color)
        ax.grid(True)

    fig.tight_layout()
    fig.show()
    
    return fig

print('now call interactively any of:')
print('fingerplot(<tau>, <delta>, "ma" or "exp", <bslen>)')
print('snrseries(<bslen>)')
print('snrmax(<bslen>)')
