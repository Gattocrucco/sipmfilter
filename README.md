# sipmfilter

This repository contains some code to study the signal/noise ratio of the SiPM output.

The scripts read files which are not included in the repository due to size. [Source](http://ds50tb.lngs.infn.it:2180/SiPM/Tiles/FBK/NUV/MB2-LF-3x/NUV-LF_3x_57/).

Scripts:

  * `fingersnr.py`: compute the SNR for moving average and exponential moving average filter varying their parameters. You have to run this script in ipython and then call functions interactively.
  
  * `fingersnrmf.py`: like `fingersnr.py` but for the matched filter.

  * `plotwav.py`: plot a pair of things to check the wav file is read correctly and makes sense.
  
  * `plotwav2.py`: a single plot for the 2020-11-03 slides.
  
  * `manine.py`: test for `fingersnr.py`.
  
  * `spectrum.py`: compute the spectra of Proto0 (the data file is not on the repository due to size) and of LNGS (wav file).
  
Modules:

  * `readwav.py`: function to read the wav files. See also sstracka/dsfe on
    bitbucket.
  
  * `integrate.py`: function to compute filters on the signals.
  
  * `fighelp.py`: convenience functions for matplotlib figures.
  
  * `single_filter_analysis.py`: make a fingerplot and compute the SNR.
  
  * `make_template.py`: make a template for the matched filter.
