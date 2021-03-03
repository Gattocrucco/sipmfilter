# sipmfilter

This repository contains some code to study filtering of the SiPM output.

## Alphabetical file index by category

### Data

The scripts read files which are not included in the repository due to size.
[First source](http://ds50tb.lngs.infn.it:2180/SiPM/Tiles/FBK/NUV/MB2-LF-3x/),
[second source](http://ds50tb.lngs.infn.it:2180/SiPM/Tiles/LFOUNDRY/pre-production-test/TILE_15/).

  * `figspectra.npz`: spectra saved by `figspectra.py`.

  * `merged_000886-adc_W201_Ch00.npz`: branch `adc_W201_Ch00` from
    `merged_000866.root`, to be loaded with `toy.DataCycleNoise`.
  
  * `nuvhd_lf_3x_tile57_77K_64V_6VoV_1-noise.npz`: noise from
    `nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav`, to be loaded with
    `toy.DataCycleNoise`.
  
  * `nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz`: template made from
    `nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav`, to be loaded with `toy.Template`.

### Scripts

  * `fig*.py`: scripts for figures in my thesis.
  
  * `fdiffrate.py`: count threshold crossings for filtered finite difference on
    a Proto0 or LNGS file.
    
  * `fdrall.py`: run `fdiffrate.py` on all files specified.
  
  * `fdrallplot.py`: plot results from `fdrall.py`.
  
  * `fingerplot.py`: do a fingerplot with an LNGS wav.

  * `fingersnr.py`: compute the filtered SNR on an LNGS wav.
  
  * `fingersnrmf.py`: **(OUTDATED)** test for `fingersnr.py`.
  
  * `hist2d.py`: do a 2D histogram of an LNGS or Proto0 file.
  
  * `hist2dall.py`: run `hist2d` on a series of files saving plots.
  
  * `lngshist.py`: **(DEPRECATED)** do a 2D histogram of an LNGS wav.

  * `manine.py`: **(OUTDATED)** test for `fingersnr.py`.
  
  * `plotwav.py`: plot a pair of things to check an LNGS wav.
  
  * `plotwav2.py`: a single plot for the 2020-11-03 slides.
  
  * `savenoise.py`: extract and save noise data from some files.
  
  * `savenoise2.py`: like above for other files.
    
  * `savetemplate.py`: compute and save a template from an LNGS wav.
  
  * `simplefilter.py`: **(OUTDATED)** signal finding.
  
  * `spectrum.py`: compute the noise spectra from some files.
  
  * `templateplot.py`: plot a template saved by `savetemplate.py`.
  
  * `toy1gsa-plot.py`: do the plots for the script below.
  
  * `toy1gsa.py`: temporal resolution at 1 GSa/s.
    
  * `toy1gvs125m-plot.py`: do the plots for the script below.

  * `toy1gvs125m.py`: comparison of temporal resolution at different sampling
    frequencies.
    
  * `toycomparison-plot.py`: do the plots for the script below.
  
  * `toycomparison.py`: temporal resolution at 125 MSa/s (also with windowing).
  
  * `toytest.py`: **(OUTDATED)** some functions to test the `toy.py` module.
      
### Modules

  * `fighelp.py`: **(DEPRECATED)** convenience functions for matplotlib figures.
  
  * `figlatex.py`: write LaTeX command for including a matplotlib figure.
    
  * `integrate.py`: filter an LNGS wav.
  
  * `make_template.py`: **(DEPRECATED)** make a template for the cross
    correlation filter from an LNGS wav.
    
  * `num2si.py`: format a number using SI suffixes.
  
  * `readroot.py`: read a Proto0 run2 root file.
 
  * `readwav.py`: read an LNGS wav; see also sstracka/dsfe on bitbucket.
  
  * `single_filter_analysis.py`: make a fingerplot and compute the SNR.
  
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
  
  * `toy1gsa/`: figures made with `toy1gsa-plot.py`.
  
  * `toy1gvs125m/`: figures made with `toy1gvs125m-plot.py`.

  * `toycomparison/`: figures made with `toycomparison-plot.py`.
  
  * `toycomparison-old/`: **(OUTDATED)** like above, but before taking a
    smaller window for filtering and localizing the signal.
    
## Dependencies

Should work with Python >= 3.6 and the standard Python scientific stack. Just
in case: developed on Python 3.8.2, required modules with version numbers are
listed in `requirements.txt`.
