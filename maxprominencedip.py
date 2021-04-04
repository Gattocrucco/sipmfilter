import numpy as np
import numba

@numba.njit(cache=True)
def maxprominencedip(events, start, end, top, n):
    """
    Find the negative peaks with the maximum prominence in arrays.
    
    For computing the prominence, maxima occuring on the border of the array
    are ignored, unless both the left and right maxima occur on the border.
        
    Parameters
    ----------
    events : array (nevents, N)
        The arrays.
    start, end : int array (nevents,)
        Each row of `events` is used only from the sample specified by
        `start` (inclusive) to `end` (exclusive).
    top : array (nevents,)
        For computing the prominence, maxima are capped at `top`.
    n : int
        The number of peaks to keep in order of prominence.
    
    Return
    ------
    position : int array (nevents, n)
        The indices of the peaks in each event, sorted along the second axis
        from lower to higher prominence. -1 for no peak found. If a local
        minimum has a flat bottom, the index of the central (rounding toward
        zero) sample is returned.
    prominence : int array (nevents, n)
        The prominence of the peaks.
    """
    
    # TODO implement using guvectorize
    
    shape = (len(events), n)
    prominence = np.full(shape, -2 ** 20, events.dtype)
    position = np.full(shape, -1)
    
    for ievent, event in enumerate(events):
        
        assert start[ievent] >= 0
        assert end[ievent] <= len(event)
        
        maxprom = prominence[ievent]
        maxprompos = position[ievent]
        relminpos = -1
        for i in range(start[ievent] + 1, end[ievent] - 1):
            
            if event[i - 1] > event[i] < event[i + 1]:
                # narrow local minimum
                relmin = True
                relminpos = i
            elif event[i - 1] > event[i] == event[i + 1]:
                # possibly beginning of wide local minimum
                relminpos = i
            elif event[i - 1] == event[i] < event[i + 1] and relminpos >= 0:
                # end of wide local minimum
                relmin = True
                relminpos = (relminpos + i) // 2
            else:
                relminpos = -1
            
            if relmin:
                # search for maximum before minimum position
                irev = relminpos
                lmax = event[irev]
                ilmax = irev
                maxmax = top[ievent]
                while irev >= start[ievent] and event[irev] >= event[relminpos] and lmax < maxmax:
                    if event[irev] > lmax:
                        lmax = event[irev]
                        ilmax = irev
                    irev -= 1
                lmax = min(lmax, maxmax)
                lmaxb = ilmax == start[ievent]
                
                # search for maximum after minimum position
                ifwd = relminpos
                rmax = event[ifwd]
                irmax = ifwd
                while ifwd < end[ievent] and event[ifwd] >= event[relminpos] and rmax < maxmax:
                    if event[ifwd] > rmax:
                        rmax = event[ifwd]
                        irmax = ifwd
                    ifwd += 1
                rmax = min(rmax, maxmax)
                rmaxb = irmax == end[ievent] - 1
                
                # compute prominence
                if (not rmaxb and not lmaxb) or (rmaxb and lmaxb):
                    maximum = min(lmax, rmax)
                elif rmaxb:
                    maximum = lmax
                elif lmaxb:
                    maximum = rmax
                prom = maximum - event[relminpos]
                
                # insert minimum into list sorted by prominence
                if prom > maxprom[0]:
                    for j in range(1, n):
                        if prom <= maxprom[j]:
                            break
                        else:
                            maxprom[j - 1] = maxprom[j]
                            maxprompos[j - 1] = maxprompos[j]
                    else:
                        j = n
                    maxprom[j - 1] = prom
                    maxprompos[j - 1] = relminpos
                
                # reset minimum flag
                relmin = False
                relminpos = -1
    
    return position, prominence

def test_maxprominencedip():
    """
    Plot a random test of `maxprominencedip`.
    """
    t = np.linspace(0, 1, 1000)
    mu = np.random.uniform(0, 1, 20)
    logsigma = np.random.randn(len(mu))
    sigma = 0.2 * np.exp(logsigma)
    wf = -np.sum(np.exp(-1/2 * ((t[:, None] - mu) / sigma) ** 2), axis=-1)
    start = 500
    pos, prom = maxprominencedip(wf[None], np.array([start]), np.array([0]), 2)
    
    fig, ax = plt.subplots(num='maxprominencedip.test_maxprominencedip', clear=True)
    
    ax.plot(wf)
    ax.axvline(start, linestyle='--')
    for i, p in zip(pos[0], prom[0]):
        print(i, p)
        if i >= 0:
            ax.vlines(i, wf[i], wf[i] + p)
            ax.axhline(wf[i] + p)
    
    fig.tight_layout()
    fig.show()

if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    
    test_maxprominencedip()
