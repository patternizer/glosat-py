![image](https://github.com/patternizer/glosat-py/blob/main/glosat-station-viewer-app.png)

# glosat-py

Experimental station viewer Plotly Python Dash app for inspecting GloSAT.p03 land surface air temperature data. Part of ongoing work for the [GloSAT Project](https://www.glosat.org):

* station info (lat, lon, elevation and WMO metadata)
* station location on zoomable OpenStreetmap
* station climate stripes ([Ed Hawkins, University of Reading](https://showyourstripes.info)
* station monthly and yearly moving average anomaly timeseries (from 1961-1990)
* station seasonal decadal mean anomaly timeseries
* station anomalies ranked by year
* station monthly climatology of absolute temperature
* station inspector app
* python reader for GloSAT.p0x station database in Pandas pickle format

[Experimental version](https://glosat-py.herokuapp.com/) available for issue-checking. 

## Contents

* `index.py` - Plotly Python Reactive Dash app index file
* `app.py` - Plotly Python dash instance
* `Procfile` - gunicorn deployment settings
* `runtime.txt` - Python build version
* `requirements.txt` - Python library dependencies
* `filter_cru_dft.py` - Python dicrete Fourier transform filter for decadal smoothing of seasonal timeseries
* `ml_optimisation.csv` - Look-up table for the DFT filter

The first step is to clone the latest glosat-py code and step into the check out directory: 

    $ git clone https://github.com/patternizer/glosat-py.git
    $ cd glosat-py

### Using Standard Python

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested 
in a conda virtual environment running a 64-bit version of Python 3.8.3.

The glosat-py app instance can be run locally at [localhost](http://127.0.0.1:8050/) by calling:

    $ python index.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)

