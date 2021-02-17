import math

# this function taken from stackoverflow and modified
# http://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6
def num2si(x, format='%.15g', si=True, space=' '):
    """
    Returns x formatted using an exponent that is a multiple of 3.

    Parameters
    ----------
    x : scalar
        The number to format.
    format : str
        Printf-style format specification used to format the mantissa.
    si : bool
        If True (default), use an SI suffix for exponent, e.g. k instead of e3,
        n instead of e-9 etc. If the exponent would be greater than 24,
        a numerical exponent is used in any case.
    space : str
        String interposed between the mantissa and the exponent.

    Returns
    -------
    fx : str
        The formatted value.

    Example
    -------
         x     | num2si(x)
    -----------|----------
       1.23e-8 |  12.3 n
           123 |  123
        1230.0 |  1.23 k
    -1230000.0 |  -1.23 M
             0 |  0
    """
    x = float(x)
    if x == 0:
        return format % x + space
    exp = int(math.floor(math.log10(abs(x))))
    exp3 = exp - (exp % 3)
    x3 = x / (10 ** exp3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = space + 'yzafpnμm kMGTPEZY'[(exp3 - (-24)) // 3]
    elif exp3 == 0:
        exp3_text = space
    else:
        exp3_text = 'e%s' % exp3 + space

    return (format + '%s') % (x3, exp3_text)
