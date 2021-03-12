def breaklines(s, maxcol=80, after='', before='', strip=True):
    pieces = [s[0]]
    for i in range(1, len(s)):
        if s[i - 1] in after or s[i] in before:
            pieces.append('')
        pieces[-1] += s[i]
    
    lines = []
    line = ''
    for p in pieces:
        if len(line) + len(p) <= maxcol:
            line += p
        else:
            lines.append(line)
            line = p
    lines.append(line)
    
    if strip:
        lines = [line.strip() for line in lines]
    
    return '\n'.join(lines)
