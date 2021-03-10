import os
import sys

import hist2d
import readroot

files = sys.argv[1:]

directory = 'hist2dall'
os.makedirs(directory, exist_ok=True)

for file in files:
    
    if '.root' in file:
        filespecs = [
            f'{file}:{tile}'
            for tile in readroot.tiles()
        ]
    else:
        filespecs = [file]
   
    for spec in filespecs:
        _, fig, _ = hist2d.main(f'{spec} -m 100000'.split())
        _, name = os.path.split(spec)
        name = name.replace(':', '_')
        savepath = f'{directory}/{name}'
        fig.savefig(savepath + '.png')
