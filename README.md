# sipmfilter

This repository contains some code to study the signal/noise ratio of the SiPM output.

The scripts read files which are not included in the repository due to size. [Source](http://ds50tb.lngs.infn.it:2180/SiPM/Tiles/FBK/NUV/MB2-LF-3x/NUV-LF_3x_57/).

Files:

  * `plotwav.py`: plot a pair of things to check the wav file is read correctly and makes sense.
  
  * `manine.py`: integrate the signals, using the trigger to locate them exactly, and plot a histogram of the results.
  
  * `spectrum.py`: compute the spectra of noise measurements (the data file is not on the repository due to size) and of events without signal (wav file).
