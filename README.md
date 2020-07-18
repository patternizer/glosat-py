# glosat-py

Python dataviz tools for GloSAT datasets:

1) plot GloSAT.prelim01 monthly anomaly timeseries
2) plot GloSAT.prelim01 mean annual anomaly timeseries
3) plot GloSAT.prelim01 mean annual anomaly 'climate stripes' (1900-2019) from 1961-1990 baseline
4) plot GloSAT.prelim01 global mean annual anomaly maps (and trial --> app) 
5) plot CRUTEM5 station level timeseries

## Contents

* `plot-prelim-stripes.py` - script to plot global, NH and SH timeseries and warming stripes from GloSAT prelim 01
* `plot-prelim-maps.py` - script to plot global maps of mean annual temperature anomaly per year from GloSAT prelim 01
* `plot-prelim-stations.py` - script to plot station level timeseries from CRUTEM5
* `plot-prelim.py` - research code 

The first step is to clone the latest glosat-py code and step into the check out directory: 

    $ git clone https://github.com/patternizer/glosat-py.git
    $ cd glosat-py
    
### Using Standard Python 

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested 
in a conda virtual environment running a 64-bit version of Python 3.6+.

glosat-py scripts can be run from sources directly, once the dependencies in the requirements.txt are resolved.

Run with:

    $ python plot-prelim-stripes.py
    $ python plot-prelim-maps.py
    $ python plot-prelim-stations.py
		        
## License

The code is distributed under terms and conditions of the [Unlicense](https://github.com/patternizer/glosat-py/blob/master/LICENSE).

## Contact information

* [Michael Taylor](https://patternizer.github.io)

