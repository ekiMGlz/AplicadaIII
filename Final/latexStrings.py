import numpy as np
import scipy.linalg as linear

def latexMatrix(M, matrixName, eq=True, complx=False, form='%0.0f', cmplxForm='%+0.0fi'):
    s=''
    if eq:
        s ='\\[ \n' + matrixName + ' = \n'
    s += '\\begin{pmatrix} \n'
    [rows, cols] = M.shape
    if complx:
        f = form+cmplxForm
        for i in range(rows):
            for j in range(cols):
                s += (f % (M[i,j].real, M[i,j].imag)) + ' '
                if not j+1 == cols:
                    s += '& '
            s += '\\\\ \n'
    else:
        for i in range(rows):
            for j in range(cols):
                s += (form % (M[i,j].real)) + ' '
                if not (j+1 == cols):
                    s += '& '
            s += '\\\\ \n'
    s += '\\end{pmatrix}'
    if eq:
        s += '\n \\]'
    return s

def latexVector(v, vecName, eq=True, complx=False, form='%0.0f', cmplxForm='%+0.0fi'):
    s=''
    if eq:
        s ='\\[ \n \\vec{' + vecName + '} = \n'
    s += '\\begin{pmatrix} \n'
    if complx:
        f = form+cmplxForm
        for x in np.nditer(v):
            s += (f % (x.real, x.imag) + ' \\\\ \n')
    else:
        for x in np.nditer(v):
            s += (form % (x.real) + ' \\\\ \n')
    s += '\\end{pmatrix}'
    if eq:
        s += '\n \\]'
    return s

def latexList(l, listName, eq=True, complx=False, form='%0.0', cmplxForm='%+0.0fi'):
    s=''
    if eq:
        s ='\\[ \n ' + listName + ' = \n'
    s += '\\{ \n'
    if complx:
        f = form+cmplxForm
        for x in l:
            s += (f % (x.real, x.imag) + ', ')
    else:
        for x in l:
            s += (form % (x.real) + ', ')
    s = s[:-2]
    s += '\\}'
    if eq:
        s += '\n \\]'
    return s


def indexedMatrix(M, row_labels, col_labels):
    [m, n] = M.shape
    s = ''
    s += '\\begin{blockarray}{c' + 'c'*n + '}\n'

    for col in col_labels:
        s += ' & ' + str(col)

    s += '\\\\'
    s += '\n\\begin{block}{c(' + 'c'*n + ')}\n'

    for i in range(m):
        s += str(row_labels[i])
        for j in range(n):
            s += " & " + str(M[i, j])
        s += " \\\\ \n"
    s += '\\end{block}\n\\end{blockarray}'
    return s
