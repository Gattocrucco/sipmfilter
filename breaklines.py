def breaklines(s, maxcol=80, after='', before='', strip=True):
    """
    Break lines in a string.
    
    Parameters
    ----------
    s : str
        The string.
    maxcol : int
        The maximum number of columns per line. It is not enforced when it is
        not possible to break the line. Default 80.
    after : str
        Characters after which it is allowed to break the line, default none.
    before : str
        Characters before which it is allowed to break the line, default none.
    strip : bool
        If True (default), remove leading and trailing whitespace from each
        line.
    
    Return
    ------
    lines : str
        The string with line breaks inserted.
    """
    pieces = [c for c in s[:1]]
    for i in range(1, len(s)):
        if s[i - 1] in after or s[i] in before:
            pieces.append('')
        pieces[-1] += s[i]
    
    lines = [p for p in pieces[:1]]
    for p in pieces[1:]:
        if len(lines[-1]) + len(p) <= maxcol:
            lines[-1] += p
        else:
            lines.append(p)
    
    if strip:
        lines = [line.strip() for line in lines]
    
    return '\n'.join(lines)
