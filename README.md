![image](https://github.com/patternizer/budyko-calculation/blob/master/glosat-station-viewer-app.png)

# glosat-py

Experimental python dataviz tools for reading, plotting and inspecting GloSAT.p0x station data from the CRUTEM land surface air temperature releases for the [GloSAT Project](https://www.glosat.org):

* python reader for GloSAT.p0x station database
* plot station level monthly timeseries
* plot station level monthly climatology
* plot station locations on world map
* plot mean annual anomaly climate stripes (1900-2019) from 1961-1990 baseline
* plot global mean annual anomaly maps
* station inspector app

[Plotly Python Reactive Dash app](https://glosat-py.herokuapp.com/) for inspecting GloSAT.p0x station timeseries, monthly climatology and location. 

## Contents

* `load-stations.py` - python reader for GloSAT.p0x station data from the CRUTEM land surface air temperature releases
* `plot-prelim-stripes.py` - script to plot global, NH and SH timeseries and climate stripes
* `plot-prelim-maps.py` - script to plot global maps of mean annual temperature anomaly per year 
* `plot-prelim-stations.py` - script to calculate baselines, anomalies and plot station level timeseries
* `app.py` - Plotly Python Reactive Dash app

The first step is to clone the latest glosat-py code and step into the check out directory: 

    $ git clone https://github.com/patternizer/glosat-py.git
    $ cd glosat-py

### Using Standard Python

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested 
in a conda virtual environment running a 64-bit version of Python 3.6+.

glosat-py scripts can be run from sources directly, once the dependencies in the requirements.txt are resolved.

Run with:

    $ python load-stations.py
    $ python plot-prelim-stripes.py
    $ python plot-prelim-maps.py
    $ python plot-prelim-stations.py
    $ python app.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)

