# sipmfilter

This repository contains some code to study filtering of the SiPM output.

## Running the code

### Software stack

Developed on Python 3.8.2 (should work with Python >= 3.6), required modules
with version numbers are listed in `requirements.txt`. To set up a python
environment you could do:

```sh
$ python3.8 -m venv python38venv
$ . python38venv/bin/activate
(python38venv) $ pip install --requirement requirements.txt
```

### Scripts

I always run the scripts in an IPython shell. If you run them with a vanilla
python interpreter the figures would close before you had a chance to look at
them (although this can be fixed by replacing `fig.show()` with `plt.show()`),
and some scripts expect you to call functions interactively after running the
script. Example in the terminal:

```
(python38venv) $ pip install ipython
(python38venv) $ ipython
Python 3.8.2 (v3.8.2:7b3ab5921f, Feb 24 2020, 17:52:18) 
Type 'copyright', 'credits' or 'license' for more information
IPython 7.13.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: %pylab
Using matplotlib backend: MacOSX
Populating the interactive namespace from numpy and matplotlib

In [2]: run myscript.py
```

### Modules

Some python files are designed to be imported. If you grab them, place them in
the same directory of your code, example: if the module file is
`ciappimodule.py`, you can import it with `import ciappimodule`.

### Data

Some scripts need large data files which I could not commit to the repository.
Some of these work without source files by using committed caches of what they
need, others can't.

There are LNGS liquid nitrogen laser test wav files and Proto0 root files. I
have received the Proto0 root files directly from someone so I can't link a
source for them. The LNGS wav files I used are found at the addresses http://ds50tb.lngs.infn.it colon 2180/SiPM/Tiles/FBK/NUV/MB2-LF-3x/ and
http://ds50tb.lngs.infn.it colon 2180/SiPM/Tiles/LFOUNDRY/pre-production-test/.

A command to automatically download all wav files in a directory could be:

```sh
$ wget --recursive --no-parent --no-directories --reject '*' --accept '*.wav' <directory url>
```

### Directories

All the scripts expect the working directory to be the root directory of the
repository. Example: instead of

```
In [1]: cd mydir

In [2]: run myscript.py
```

You should do:

```
In [1]: run mydir/myscript.py
```

They also expect to find the large data files in a directory named `darksidehd`.
The scripts in `figthesis/` save figures in `../thesis/figures`.

## Alphabetical file index by category

### Data

  * `DS_proto_runs_nov_2019.csv`: Proto0 metadata, dated February 2021.

  * `figthesis/figspectra.npz`: spectra saved by `figspectra.py`.

  * `noises/merged_000886-adc_W201_Ch00.npz`: branch `adc_W201_Ch00` from
    `merged_000866.root`, to be loaded with `toy.DataCycleNoise`.
  
  * `noises/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-noise.npz`: noise from
    `nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav`, to be loaded with
    `toy.DataCycleNoise`.
  
  * `templates/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz`: template made
    from `nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav`, to be loaded with
    `template.Template`.

### Scripts

  * `afterpulse_tile15_ap.py`: measure dcr and afterpulse in tile 15.

  * `afterpulse_tile15.py`: **(OUTDATED)** measure dcr in tile 15.
  
  * `afterpulse_tile21.py`: measure dcr and afterpulse in tile 21.

  * `afterpulse_tile57.py`: count afterpulses in tile 57.

  * `figthesis/fig*.py`: scripts for figures in my thesis.
  
  * `fdiffrate.py`: count threshold crossings for filtered finite difference on
    a Proto0 or LNGS file.
    
  * `fdrall.py`: run `fdiffrate.py` on all files specified.
  
  * `fdrallplot.py`: plot results from `fdrall.py`.
  
  * `fingerplot_tile21.py`: a quick fingerplot of tile 21 files.
  
  * `fingerplot.py`: do a fingerplot with an LNGS wav.

  * `fingersnr.py`: compute the filtered SNR on an LNGS wav.
  
  * `fingersnrmf.py`: **(OUTDATED)** test for `fingersnr.py`.
  
  * `hist2d.py`: do a 2D histogram of an LNGS or Proto0 file.
  
  * `hist2dall.py`: run `hist2d` on a series of files saving plots.
  
  * `lngshist.py`: **(DEPRECATED)** do a 2D histogram of an LNGS wav.

  * `manine.py`: **(OUTDATED)** test for `fingersnr.py`.
  
  * `plotevent.py`: plot a single event from a Proto0 or LNGS file.
  
  * `plotwav.py`: plot a pair of things to check an LNGS wav.
  
  * `plotwav2.py`: a single plot for the 2020-11-03 slides.
  
  * `savenoise.py`: extract and save noise data from some files.
  
  * `savenoise2.py`: like above for other files.
    
  * `savetemplate.py`: compute and save a template from an LNGS wav.
  
  * `simplefilter.py`: **(OUTDATED)** signal finding.
  
  * `spectrum.py`: compute the noise spectra from some files.
  
  * `templateplot.py`: plot a template saved by `savetemplate.py`.
  
  * `tile21gap.py`: determine the size of events in tile 21 files.
  
  * `toy1gsa-plot.py`: do the plots for the script below.
  
  * `toy1gsa.py`: temporal resolution at 1 GSa/s.
    
  * `toy1gvs125m-plot.py`: do the plots for the script below.

  * `toy1gvs125m.py`: comparison of temporal resolution at different sampling
    frequencies.
    
  * `toycomparison-plot.py`: do the plots for the script below.
  
  * `toycomparison.py`: temporal resolution at 125 MSa/s (also with windowing).
  
  * `toytest.py`: **(OUTDATED)** some functions to test the `toy.py` module.
  
  * `triggerhist.py`: histogram the trigger leading edge position.
      
### Modules

  * `afterpulse.py`: class to analyze an LNGS file.
  
  * `argminrelmin.py`: index of the minimum local minimum.
  
  * `breaklines.py`: function to break lines in a string.
  
  * `correlate.py`: compute the cross correlation.
  
  * `downcast.py`: downcast a numpy data type recursively.

  * `fighelp.py`: **(DEPRECATED)** convenience functions for matplotlib figures.
  
  * `figlatex.py`: write LaTeX command for including a matplotlib figure.
  
  * `firstbelowthreshold.py`: index of first array element below a threshold.
    
  * `integrate.py`: filter an LNGS wav.
  
  * `make_template.py`: **(DEPRECATED)** make a template for the cross
    correlation filter from an LNGS wav.
    
  * `maxprominencedip.py`: find max prominence negative peaks.
  
  * `meanmedian.py`: mean of the median over strided subarrays.
  
  * `npzload.py`: class to serialize objects to numpy archives.
    
  * `num2si.py`: format a number using SI suffixes.
  
  * `poissonsup.py`: compute upper confidence limits for the poisson.
  
  * `read.py`: read a Proto0 or LNGS file with a single interface.
  
  * `readroot.py`: read a Proto0 run2 root file.
 
  * `readwav.py`: read an LNGS wav; see also sstracka/dsfe on bitbucket.
  
  * `runsliced.py`: do something in batches with a progressbar.
  
  * `single_filter_analysis.py`: make a fingerplot and compute the SNR.
  
  * `template.py`: class to make signal templates.
  
  * `textbox.py`: write text in a box on plots.
  
  * `textmatrix.py`: format tables for LaTeX.
  
  * `toy.py`: simulation to measure temporal resolution.

### Output

  * `figures/`: miscellaneous figures made with the scripts.
  
  * `fingersnr-changebaseline/`: **(OUTDATED)** the output from `fingersnr.py`
    for the 2020-11-03 slides.
  
  * `fingersnr-changebaseline-mf/`: the same after adding the matched filter.
  
  * `fingersnr-mftemplate/`: plot of the matched filter template and its
    spectrum.
  
  * `min_snr_ratio.py`: **(OUTDATED)** saved ratios filtered SNR/unfiltered SNR.
  
  * `slides-2020-11-03/`: presentation on the filtered SNR.
  
  * `slides-2020-12-01/`: presentation on the temporal resolution.
  
  * `slides-2020-12-10/`: presentation on the effect of waveform truncation.
  
  * `slides-2021-03-11/`: presentation on DCR and afterpulses of LF_TILE15. 
  
  * `toy1gsa/`: figures made with `toy1gsa-plot.py`.
  
  * `toy1gvs125m/`: figures made with `toy1gvs125m-plot.py`.

  * `toycomparison/`: figures made with `toycomparison-plot.py`.
  
  * `toycomparison-old/`: **(OUTDATED)** like above, but before taking a
    smaller window for filtering and localizing the signal.
