"""
Compute the filtered SNR on an LNGS wav.

The file to read is hardcoded at the bottom of this script. Run this script in
an IPython shell and then call functions as suggested by the onscreen
instructions.

This script can be loaded as a module to use the functions `snrplot`,
`snrmaxplot` and `snrmaxplot_multiple`. Other functions may not work.

Functions
---------
make_tau_delta :
    Generate a reasonable range of delays from trigger for filter evaluation
    for a list of filter lengths.
snrseries :
    Compute the SNR for a range of filter length and delay from trigger.
snrplot :
    Plot the output of `snrseries`.
templateplot :
    Plot the matched filter template.
fingerplot :
    Plot a fingerplot for a chosen filter.
snrmax :
    Find the delay from trigger that maximizes the SNR.
snrmaxplot :
    Plot the output of `snrmax`.
snrmaxplot_multiple :
    Plot together the outputs of multiple `snrmax` invocations.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import optimize
import tqdm

import readwav
import integrate
from single_filter_analysis import single_filter_analysis
from make_template import make_template

def make_tau_delta(tau, ndelta, flat=True):
    """
    Make a range of delta (offset from trigger) for each tau (length parameter
    of the filter) for each filter. The filters are
    
        "ma" moving average,
    
        "exp" exponential moving average,
    
        "mf" matched filter.
    
    The output is meant to be used as arguments to integrate.filter().
    
    Parameters
    ----------
    tau : array (ntau,)
        Values of the length parameter.
    ndelta : int
        Number of delta values in each range.
    flat : bool
        If True, return 1D arrays, else (ntau, ndelta).
    
    Return
    ------
    tau, delta_ma, delta_exp, delta_mf : int arrays
        The shape is (ntau, ndelta) if flat=False else (ntau * ndelta,).
    """
    
    # make tau same shape as delta
    tau = np.broadcast_to(tau.reshape(-1, 1), (len(tau), ndelta))

    # delta for moving average
    delta_ma_rel = np.linspace(0.5, 1.4, ndelta)
    delta_ma_off = 80 + np.linspace(-40, 40, ndelta)
    taueff_ma = 500 * (tau / 500) ** (4/5)
    delta_ma = delta_ma_off + delta_ma_rel * taueff_ma

    # delta for exponential moving average
    delta_rel_exp = np.linspace(0.1, 2, ndelta)
    delta_off_exp = np.linspace(65, 400, ndelta)
    taueff_exp = 512 * (tau / 512) ** (3/5)
    delta_exp = delta_off_exp + delta_rel_exp * taueff_exp

    # delta for matched filter
    delta_off_mf = 10 * (np.arange(ndelta) - ndelta // 2)
    delta_mf = delta_off_mf + tau
    
    # convert to int and reshape
    arrays = ()
    for x in [tau, delta_ma, delta_exp, delta_mf]:
        x = np.array(np.rint(x), int)
        if flat:
            x = x.reshape(-1)
        arrays += (x,)
    
    return arrays

_default_tau = np.array([32, 64, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048])
_default_ndelta = 10

def snrseries(tau=_default_tau, ndelta=_default_ndelta, bslen=8000, plot=True):
    """
    Compute SNR as a function of tau and delta. Make a plot and return the
    results.
    
    Parameters
    ----------
    tau : array (ntau,)
        Length parameter of the filters. 
    ndelta : int
        Number of values of offset from trigger explored in a hardcoded range.
    bslen : int
        The number of samples used for the baseline.
    plot : bool
        If False, do not plot. The plot can be done separately by calling
        snrplot().
    
    Returns
    -------
    tau : array (ntau,)
        Values of the filter scale parameter.
    delta_ma : array (ntau, ndelta)
        Values of the offset for the moving average for each tau.
    delta_exp : array (ntau, ndelta)
        Values of the offset for the exponential moving average for each tau.
    delta_mf : array (ntau, ndelta)
        Values of the offset for the matched filter for each tau.
    waveform : array (max(tau),)
        Template used for the matched filter.
    snr : array (3, ntau, ndelta)
        The SNR for (moving average, exponential moving average, matched
        filter), and for each length parameter (tau) and offset from trigger
        (delta).
    """
    
    # Generate delta ranges.
    ntau = len(tau)
    tau, delta_ma, delta_exp, delta_mf = make_tau_delta(tau, ndelta, flat=True)
    
    print('make template for matched filter...')
    w0 = make_template(data, ignore, np.max(tau) + 200, noisecorr=False)
    start_mf = integrate.make_start_mf(w0, tau)
    # waveform = make_template(data, ignore, np.max(tau + start_mf), noisecorr=True)
    waveform = w0
    
    print('computing filters...')
    start, baseline, vma, vexp, vmf = integrate.filter(data, bslen, delta_ma, tau, delta_exp, tau, delta_mf, waveform, tau, start_mf)

    snr = np.empty((3, len(tau)))

    print('analysing filter output...')
    for i in tqdm.tqdm(range(snr.shape[1])):
        for j, value in enumerate([vma, vexp, vmf]):
            value = value[:, i]
            corr_value = (baseline - value)[~ignore]
            snr[j, i] = single_filter_analysis(corr_value)
    
    # Reshape arrays, make plot and return.
    output = (tau.reshape(ntau, ndelta)[:, 0],)
    for x in [delta_ma, delta_exp, delta_mf]:
        output += (x.reshape(ntau, ndelta),)
    output += (waveform, snr.reshape(-1, ntau, ndelta))
    if plot:
        snrplot(*output)
    return output

def snrplot(tau, delta_ma, delta_exp, delta_mf, waveform, snr, fig1=None, fig2=None, plottemplate=True):
    """
    Plot SNR as a function of tau and delta. Called by snrseries().
    
    Parameters
    ----------
    tau, delta_ma, delta_exp, delta_mf, waveform, snr : arrays
        The output from snrseries().
    fig1, fig2 : matplotlib figure, optional
        The figures where the plot is drawn.
    plottemplate : bool
        If True (default), plot the matched filter template.
    
    Returns
    -------
    fig1, fig2 : matplotlib figure
        The figures with the plots.
    """
    
    if fig1 is None:
        fig = plt.figure('fingersnr-snrplot', figsize=[10.2, 7.1])
        fig.clf()
    else:
        fig = fig1
    
    grid = gridspec.GridSpec(2, 2)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1], sharex=ax0, sharey=ax0)
    ax2 = fig.add_subplot(grid[1, :], sharex=ax0, sharey=ax0)
    axs = [ax0, ax1, ax2]

    axs[0].set_title('Moving average')
    axs[1].set_title('Exponential moving average')
    axs[2].set_title('Cross correlation')
    
    for i, (ax, d) in enumerate(zip(axs, [delta_ma, delta_exp, delta_mf])):
        for j in range(len(tau)):
            alpha = 1 - (j / len(tau))
            label = f'{tau[j]}'
            ax.plot(d[j], snr[i, j], color='black', alpha=alpha, label=label)
        if ax.is_first_col():
            ax.set_ylabel('SNR')
        if ax.is_last_row():
            ax.set_xlabel('Offset from trigger [ns]')
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
    
    axs[2].legend(loc='best', title='Filter length [ns]', ncol=2)

    fig.tight_layout()
    fig.show()
    
    fig1 = fig
    
    if plottemplate and fig2 is None:
        fig = plt.figure('fingersnr-snrplot2')
        fig.clf()
    else:
        fig = fig2
    
    if plottemplate:
        ax = fig.subplots(1, 1)
        ax.set_title('Matched filter template')
        ax.set_xlabel('Sample number [ns]')
        ax.plot(waveform)
        ax.grid()
    
        fig.tight_layout()
        fig.show()
    
    fig2 = fig
    
    return fig1, fig2

def templateplot(n=2048):
    """
    Compute the template for the matched filter and plot it.
    
    Parameters
    ----------
    n : int
        Length of the template. The template starts with the trigger.

    Return
    ------
    fig1, fig2 : matplotlib figures
    """
    
    fig1 = plt.figure('fingersnr-templateplot-1')
    fig2 = plt.figure('fingersnr-templateplot-2')
    fig1.clf()
    fig2.clf()
    
    make_template(data, ignore, n, True,  fig1, fig2)
    
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.show()
    fig2.show()
    
    return fig1, fig2

def fingerplot(tau, delta, kind='ma', bslen=8000):
    """
    Make a fingerplot with a specific filter and print the SNR.
    
    Parameters
    ----------
    tau : int
        Length parameter of the filter.
    delta : int
        Offset from the trigger where the filter is evaluated.
    kind : str
        One of 'ma' = moving average, 'exp' = exponential moving average,
        'mf' = matched filter, 'mfn' = matched filter with noise correction.
    bslen : int
        Number of samples used for the baseline.
    
    Return
    ------
    fig1, fig2 : matplotlib figures
    """
    
    if kind == 'ma':
        start, baseline, value = integrate.filter(data, bslen, delta_ma=delta, length_ma=tau)
    elif kind == 'exp':
        start, baseline, value = integrate.filter(data, bslen, delta_exp=delta, tau_exp=tau)
    elif kind in ('mf', 'mfn'):
        w0 = make_template(data, ignore, tau + 200, noisecorr=False)
        start_mf = integrate.make_start_mf(w0, tau)
        if kind == 'mfn':
            waveform = make_template(data, ignore, tau + start_mf[0], noisecorr=True)
        else:
            waveform = w0
        start, baseline, value = integrate.filter(data, bslen, delta_mf=delta, waveform_mf=waveform, length_mf=tau, start_mf=start_mf)
    else:
        raise KeyError(kind)
    
    corr_value = (baseline - value[:, 0])[~ignore]
    fig1 = plt.figure('fingersnr-fingerplot-1', figsize=[7.27, 5.73])
    fig2 = plt.figure('fingersnr-fingerplot-2', figsize=[6.4, 4.8])
    fig1.clf()
    fig2.clf()
    snr = single_filter_analysis(corr_value, fig1, fig2)
    print(f'snr = {snr:.2f}')
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.show()
    fig2.show()
    
    return fig1, fig2
    
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

def snrmax(tau=_default_tau, bslen=8000, plot=True, hint_delta_ma=None):
    """
    Find the maximum SNR varying delta for each tau. "Delta" is the offset from
    the trigger. Also plot the results.
    
    Parameters
    ----------
    tau : array (ntau,)
        Values of the length parameter of the filters.
    bslen : int
        The number of samples used for the baseline.
    plot : bool
        If False, do not plot. Use snrmaxplot() separately.
    hint_delta_ma : array (ntau,), optional
        A guess on the maximum position for the moving average.
    
    Returns
    -------
    tau : array (ntau,)
        The tau values tested.
    snrmax : array (3, ntau)
        The maximum SNR for each tau, first dimension is (moving average,
        exponential moving average, matched filter).
    deltarange : array (3, ntau, 3)
        The triplets [delta_left, delta_max, delta_right] where delta_max is
        the delta that maximizes the SNR and delta_left and _right are points
        where the SNR is -1 relative to the maximum.
    
    """
    # Function to be minimized, returns -snr.
    def fun(delta, tau, kind, waveform, start_mf):
        try:
            if kind == 'exp':
                start, baseline, value = integrate.filter(data, bslen, delta_exp=delta, tau_exp=tau)
            elif kind == 'ma':
                start, baseline, value = integrate.filter(data, bslen, delta_ma=delta, length_ma=tau)
            elif kind == 'mf':
                start, baseline, value = integrate.filter(data, bslen, delta_mf=delta, length_mf=tau, waveform_mf=waveform, start_mf=start_mf)
            else:
                raise KeyError(kind)
        except ZeroDivisionError:
            return 0
        corr_value = (baseline - value[:, 0])[~ignore]
        snr = single_filter_analysis(corr_value)
        return -snr
    
    ntau = len(tau)
    
    print('make template for matched filter...')
    waveform = make_template(data, ignore, np.max(tau) + 200)
    start_mf = integrate.make_start_mf(waveform, tau)

    print('maximizing SNR for each tau...')
    snrmax = np.full((3, ntau), np.nan)
    deltarange = np.full((3, ntau, 3), np.nan)
    # dim0: MOVAVG, EXPAVG, MATFIL
    # dim1: tau
    # dim2: left, max, right
    for i in tqdm.tqdm(range(ntau)):
        t = tau[i]
        for j, kind in enumerate(['ma', 'exp', 'mf']):
            args = (t, kind, waveform, start_mf)
            bracket = (66 + t * 0.8, 66 + t * 1.2)
            if kind == 'mf':
                bracket = (t - 20, t, t + 20)
            elif kind == 'ma' and hint_delta_ma is not None:
                c = hint_delta_ma[i]
                bracket = (c, 1.1 * c)
            options = dict(xtol=1, maxiter=20)
            kw = dict(bracket=bracket, args=args, options=options, method='golden')
            try:
                result = optimize.minimize_scalar(fun, **kw)
                if not result.success:
                    print(f'i={i}, j={j}, max: {result}')
                deltamax = result.x
                deltarange[j, i, 1] = deltamax
                snrmax[j, i] = -result.fun
            except ValueError: # "Not a bracketing interval."
                continue
        
            f = lambda *args: fun(*args) - (1 - snrmax[j, i])
            kw = dict(args=args, options=options, method='bisect')
            
            try:
                bracket = (0, deltamax)
                result = optimize.root_scalar(f, bracket=bracket, **kw)
                if not result.converged:
                    print(f'i={i}, j={j}, left: {result}')
                deltarange[j, i, 0] = result.root
            except ValueError: # "f(a) and f(b) must have different signs"
                pass
            
            try:
                bracket = (deltamax, 3 * deltamax)
                result = optimize.root_scalar(f, bracket=bracket, **kw)
                if not result.converged:
                    print(f'i={i}, j={j}, right: {result}')
                deltarange[j, i, 2] = result.root
            except ValueError:
                pass
    
    output = (tau, snrmax, deltarange)
    if plot:
        snrmaxplot(*output)
    return output

def snrmaxplot(tau, snrmax, deltarange, fig=None, plotoffset=True):
    """
    Plot the output from snrmax(). Called by snrmax().
    
    Parameters
    ----------
    tau, snrmax, deltarange : array
        The things returned by snrmax().
    fig : matplotlib figure, optional
        The figure where the plot is drawn.
    plotoffset : bool
        If True (default), plot the offset from trigger that maximizes the SNR.
    
    Returns
    -------
    fig : matplotlib figure
    """
    if fig is None:
        fig = plt.figure('fingersnr-snrmaxplot', figsize=[6.4, 7.1])
        fig.clf()
    
    if plotoffset:
        ax0, ax1, ax2 = fig.subplots(3, 1, sharex=True)
    else:
        ax0, ax2 = fig.subplots(2, 1, sharex=True)
        ax1 = None

    _snrmaxplot_core(tau, snrmax, deltarange, ax0, ax1, ax2)
    
    fig.tight_layout()
    fig.show()
    
    return fig

def snrmaxplot_multiple(fig, snrmaxout):
    """
    Plot the output from multiple snrmax() invocations.
    
    Parameters
    ----------
    fig : matplotlib figure, optional
        The figure where the plot is drawn.
    snrmaxout : list of tuples
        The output(s) from snrmax.
    
    Return
    ------
    axs : matplotlib axes
        A 2 x len(snrmaxout) array of axes.
    """
    axs = fig.subplots(2, len(snrmaxout), sharex=True, sharey='row')
    
    for i, (ax0, ax2) in enumerate(axs.T):
        _snrmaxplot_core(*snrmaxout[i], ax0, None, ax2, legendkw=dict(fontsize='small', title_fontsize='medium'))
    
    return axs

def _snrmaxplot_core(tau, snrmax, deltarange, ax0, ax1, ax2, legendkw={}):
    if ax0.is_first_col():
        ax0.set_ylabel('Maximum SNR')
    if ax1 is not None and ax1.is_first_col():
        ax1.set_ylabel('Offset from trigger\nthat maximizes the SNR [ns]')
    if ax2.is_first_col():
        ax2.set_ylabel('Width of maximum\n of SNR vs. offset [ns]')
    ax2.set_xlabel('Filter length [ns]')
    
    kws = {
        'moving average'            : dict(linestyle='-', color='black', marker='x'),
        'exponential moving average': dict(linestyle='--', color='black', marker='.'),
        'cross correlation'         : dict(linestyle=':', color='black', marker='o', markerfacecolor='white'),
    }
    
    for i, (label, kw) in enumerate(kws.items()):
        x = tau + 12 * (i - 1)

        ax0.plot(tau, snrmax[i], label=label, **kw)
    
        dr = deltarange[i].T
        if ax1 is not None:
            sel = snrmax[i] > 0
            ax1.plot(tau[sel], dr[1, sel], **kw)
            # yerr = np.array([
            #     dr[1] - dr[0],
            #     dr[2] - dr[1]
            # ])
            # ax.errorbar(x[sel], dr[1, sel], yerr=yerr[:, sel], fmt='.', color=color, capsize=4)
        
        ax2.plot(tau, dr[2] - dr[0], **kw)
    
    if ax0.is_first_col():
        ax0.legend(title='Filter', **legendkw)
    for ax in [ax0, ax1, ax2]:
        if ax is not None:
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')
    if ax1 is not None:
        ax1.set_yscale('log')
    ax2.set_yscale('log')

if __name__ == '__main__':
    # Load wav file.
    filename = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
    data = readwav.readwav(filename, mmap=False)
    ignore = readwav.spurious_signals(data)
    print(f'ignoring {np.sum(ignore)} events with signals in baseline zone')

    print('now call interactively any of:')
    print('fingerplot(<tau>, <delta>, "ma" or "exp" or "mf", <bslen>)')
    print('templateplot(<length>)')
    print('snrseries(<taus>, <ndelta>, <bslen>)')
    print('snrmax(<taus>, <bslen>, hint_delta_ma=<approx delta for max SNR>)')
