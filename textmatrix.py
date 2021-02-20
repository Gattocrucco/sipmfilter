def _array_like(obj):
    return hasattr(obj, '__len__')

class TextMatrix(object):
    """
    Object to format tables.
    
    Methods
    -------
    text : generic formatter.
    latex : format as latex table.
    transpose : transpose the matrix.
    __mul__ : multiplication stacks matrices horizontally.
    __truediv__ : division stacks matrices vertically.
    """
    
    def __init__(self, matrix, fill='', fill_side='right', fillrows=True):
        """
        Create a 2D matrix of arbitrary objects.
        
        Parameters
        ----------
        matrix : tipically list of lists
            An object that can be interpreted as 2D matrix. If it can not,
            the resulting TextMatrix is 1x1. If it can be interpreted as
            a 1D array, then each row has length 1.
        fill : object
            If the rows are uneven, shorter ones are filled with this object.
        fill_side : {'right', 'left'}
            Specify on which side shorter rows are filled.
        fillrows : bool
            Default True. If False, let rows be uneven.
        """
        # make sure matrix is at least 1D
        if not _array_like(matrix):
            matrix = [matrix]
        matrix = list(matrix)
        
        # make sure each element of matrix is at least 1D
        for i in range(len(matrix)):
            if not _array_like(matrix[i]):
                matrix[i] = [matrix[i]]
            matrix[i] = list(matrix[i])
            
        # make sure each element of matrix has the same length
        maxlength = max(map(len, matrix), default=0)
        for i in range(len(matrix)):
            if len(matrix[i]) != maxlength and fillrows:
                if fill_side == 'right':
                    matrix[i] = list(matrix[i]) + [fill] * (maxlength - len(matrix[i]))
                elif fill_side == 'left':
                    matrix[i] = [fill] * (maxlength - len(matrix[i])) + list(matrix[i])
                else:
                    raise KeyError(fill_side)
        
        self._matrix = matrix

    def __repr__(self):
        return self.text(before=' ')
    
    def __str__(self):
        return self.text(before=' ')
    
    def text(self, before='', after='', between='', newline='\n', subs={}):
        """
        Format the matrix as a string. Each element in the matrix
        is converted to a string using str(), then elements are concatenated
        in left to right, top to bottom order.
        
        Parameters
        ----------
        before, after, between : string
            Strings placed respectively before, after, between the elements.
        newline : string
            String placed after each row but the last.
        subs : dictionary
            Dictionary specifying substitutions applied to each element,
            but not to the parameters <before>, <after>, etc. The keys
            of the dictionary are the strings to be replaced, the values
            are the replacement strings. If you want the substitutions
            to be performed in a particular order, use a OrderedDict
            from the <collections> module.
        
        Returns
        -------
        s : string
            Matrix formatted as string.
        """
        if len(self._matrix) == 0:
            return ''
        
        # convert matrix elements to strings, applying subs
        str_matrix = []
        for row in self._matrix:
            str_row = []
            for element in row:
                element = str(element)
                for sub, rep in subs.items():
                    element = element.replace(sub, rep)
                str_row.append(element)
            str_matrix.append(str_row)
        
        # get maximum width of each column
        maxncols = max(map(len, str_matrix), default=0)
        length_T = [[] for _ in range(maxncols)]
        for str_row in str_matrix:
            for icol, element in enumerate(str_row):
                length_T[icol].append(len(element))
        colwidth = list(map(lambda x: max(x, default=0), length_T))
        
        # convert string matrix to text
        s = ''
        nrows = len(str_matrix)
        for irow, str_row in enumerate(str_matrix):
            ncols = len(str_row)
            for icol, element in enumerate(str_row):
                formatter = '{:>%d}' % colwidth[icol]
                s += before + formatter.format(element) + after
                if icol < ncols - 1:
                    s += between
            if irow < nrows - 1:
                s += newline
        
        return s
    
    def latex(self, **kwargs):
        """
        Format the matrix as a LaTeX table.
        
        Keyword arguments
        -----------------
        Keyword arguments are passed to the <text> method, taking
        precedence on settings for LaTeX formatting. The `subs` argument is
        treated separately: the default one is updated with the contents of the
        one from **kwargs.
        
        Returns
        -------
        s : string
            Matrix formatted as LaTeX table.
        """
        subs = {
            '%': '\\%',
            '&': '\\&',
        }
        kw = dict(before='', after='', between=' & ', newline=' \\\\\n', subs=subs)
        kw['subs'].update(kwargs.pop('subs', dict()))
        kw.update(kwargs)
        return self.text(**kw)
    
    def transpose(self):
        """
        Returns a transposed copy of the matrix. The elements are not copied.
        """
        return TextMatrix([
            [
                self._matrix[row][col]
                for row in range(len(self._matrix))
            ] for col in range(len(self._matrix[0]))
        ])
    
    def __mul__(self, other):
        """Multiplication concatenates two matrices horizontally."""
        if not isinstance(other, TextMatrix):
            return NotImplemented
        assert len(other._matrix) == len(self._matrix)
        return TextMatrix([l + r for l, r in zip(self._matrix, other._matrix)], fill=None)
    
    def __truediv__(self, other):
        """Division concatenates two matrices vertically."""
        if not isinstance(other, TextMatrix):
            return NotImplemented
        return TextMatrix(self._matrix + other._matrix, fill=None)
