# sipmfilter

This repository contains some code to study filtering of the SiPM output.

## Data

The scripts read files which are not included in the repository due to size. [Source](http://ds50tb.lngs.infn.it:2180/SiPM/Tiles/FBK/NUV/MB2-LF-3x/).

  * `merged_000866.npy`: branch `adc_W201_Ch00` from `merged_000866.root` (the root file is not in the repo).
  
  * `toycomparison-lngs.npy`: noise from `nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav` downsampled to 125 MSa/s.

## Scripts

  * `toycomparison.py`: a comparison of time resolution for various filters.

  * `fingersnr.py`: compute the SNR for moving average, exponential moving
    average and matched filter varying their parameters. You have to run this
    script in ipython and then call functions interactively.
  
  * `toytest.py`: some functions to test the `toy.py` module.
  
  * `fingersnrmf.py`: test for `fingersnr.py`.

  * `manine.py`: test for `fingersnr.py`.
  
  * `plotwav.py`: plot a pair of things to check the wav file is read correctly
    and makes sense.
  
  * `plotwav2.py`: a single plot for the 2020-11-03 slides.
  
  * `spectrum.py`: compute the spectra of Proto0 (the data file is not on the
    repository due to size) and of LNGS (wav file).
  
  * `simplefilter.py`: signal finding. Probably does not work anymore with the
    rest of the code.
  
## Modules

  * `toy.py`: classes and functions to do a Monte Carlo to measure time
    localization resolution.

  * `readwav.py`: function to read the wav files. See also sstracka/dsfe on
    bitbucket.
  
  * `integrate.py`: function to filter the LNGS wav files.
  
  * `single_filter_analysis.py`: make a fingerplot and compute the SNR.
  
  * `make_template.py`: make a template for the matched filter. Outdated,
    contains an attempt at matching the noise spectrum (has numerical
    accuracy issues).
 
  * `fighelp.py`: convenience functions for matplotlib figures.
  
## Output

  * `fingersnr-changebaseline/`: the output from `fingersnr.py` for the 2020-11-03 slides.
  
  * `fingersnr-changebaseline-mf/`: the same after adding the matched filter.
  
  * `fingersnr-mftemplate/`: plot of the matched filter template and its spectrum.
  
  * `min_snr_ratio.py`: saved ratios filtered SNR/unfiltered SNR.
  
  * `slides-2020-11-03/`: presentation on the filtered SNR.
  
  * `slides-2020-12-01/`: presentation on the temporal localization resolution.
  
  * `figures/`: miscellaneous figures made with the scripts.
  
  * `toycomparison/`: figures made with `toycomparison.py`.

## Dependencies

Should work with Python >= 3.6 and the standard Python scientific stack. Just in case: developed on Python 3.8.2, required modules with version numbers are listed in `requirements.txt`.
