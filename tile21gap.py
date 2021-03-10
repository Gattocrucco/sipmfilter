import os
import glob

import numpy as np

files = glob.glob('darksidehd/*TILE21*.wav')
filesizes = [
    os.path.getsize(file) for file in files
]

sizes = np.unique(filesizes)
differences = np.diff(sizes)
eventsize = np.gcd.reduce(differences)

print(sizes)
print(differences)
print(eventsize)
