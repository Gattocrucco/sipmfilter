def textbox(ax, text, loc='lower left', **kw):
    """
    Draw a box with text on a matplotlib plot.
    
    Parameters
    ----------
    ax : matplotlib axis
        The plot where the text box is drawn.
    text : str
        The text.
    loc : str
        The location of the box. Format: 'lower/center/upper left/center/right'.
    
    Keyword arguments
    -----------------
    Additional keyword arguments are passed to ax.annotate. If you pass a
    dictionary for the `bbox` argument, the defaults are updated instead of
    resetting the bounding box properties.
    
    Return
    ------
    The return value is that from ax.annotate.
    """
    
    M = 8
    locparams = {
        'lower left'   : dict(xy=(0  , 0  ), xytext=( M,  M), va='bottom', ha='left'  ),
        'lower center' : dict(xy=(0.5, 0  ), xytext=( 0,  M), va='bottom', ha='center'),
        'lower right'  : dict(xy=(1  , 0  ), xytext=(-M,  M), va='bottom', ha='right' ),
        'center left'  : dict(xy=(0  , 0.5), xytext=( M,  0), va='center', ha='left'  ),
        'center center': dict(xy=(0.5, 0.5), xytext=( 0,  0), va='center', ha='center'),
        'center right' : dict(xy=(1  , 0.5), xytext=(-M,  0), va='center', ha='right' ),
        'upper left'   : dict(xy=(0  , 1  ), xytext=( M, -M), va='top'   , ha='left'  ),
        'upper center' : dict(xy=(0.5, 1  ), xytext=( 0, -M), va='top'   , ha='center'),
        'upper right'  : dict(xy=(1  , 1  ), xytext=(-M, -M), va='top'   , ha='right' ),
    }
    
    kwargs = dict(
        fontsize='x-small',
        xycoords='axes fraction',
        textcoords='offset points',
        bbox=dict(
            facecolor='white',
            alpha=0.75,
            edgecolor='#ccc',
            boxstyle='round'
        ),
    )
    kwargs.update(locparams[loc])
    
    newkw = dict(kw)
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(kwargs.get(k, None), dict):
            kwargs[k].update(v)
            newkw.pop(k)
    kwargs.update(newkw)
    
    return ax.annotate(text, **kwargs)
