# glosat-py

Python dataviz tools for GloSAT datasets:

1) plot GloSAT.prelim01 monthly anomaly timeseries
2) plot GloSAT.prelim01 mean annual anomaly timeseries
3) plot GloSAT.prelim01 mean annual anomaly 'climate stripes' (1900-2019) from 1961-1990 baseline
4) plot GloSAT.prelim01 global mean annual anomaly maps (and trial --> app) 

## Contents

* `plot-prelim.py` - main script to be run with Python 3.6+

The first step is to clone the latest glosat-py code and step into the check out directory: 

    $ git clone https://github.com/patternizer/glosat-py.git
    $ cd glosat-py
    
### Using Standard Python 

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested 
in a conda virtual environment running a 64-bit version of Python 3.6+.

glosat-py scripts can be run from sources directly, once the dependencies in the requirements.txt are resolved.

Run with:

    $ python plot-prelim.py
	        
## License

The code is distributed under terms and conditions of the [Unlicense](https://github.com/patternizer/glosat-py/blob/master/LICENSE).

## Contact information

* [Michael Taylor](https://patternizer.github.io)

