Han-Kwang's Covid-19 repository.

Author: Han-Kwang Nienhuys
Email: not disclosed.
License: MIT

Here are a few bits of Python code for epidemiological simulations
and analysis of NL statistics.

I develop using Anaconda Python (version 2020.02) in Linux. The code
should generally run in Windows and other Python distributions, but is
not tested. See requirements.txt for the package versions. There was a
report of errors with a different pandas version; the other packages
are less critical (I think).

Here are the Python files:

  casus_analysis.py - module for parsing/analyzing NL casus data.
  chen_airborne_exposure_plot.py - analysis of data on droplet range.
  holiday_regions.py - mapping municipalities to school-holiday regions
  nlcovidstats.py - download and plot daily number of positives.
  process_casus_data.py - script demonstrating casus analysis.
  run_seirmodel.py - script demonsrating the SEIR model.
  seirmodel.py - SEIR epidimiological model (not maintained).
  test_seirmodel.py - test cases for SEIR model
  tools.py - helper functions.

In order to use casus_analysis/process_casus_data, you need to
download the archive of casus data. In Linux:

  cd data-casus
  wget https://github.com/mzelst/covid-19/archive/master.zip
  unzip "covid-19-master/data-rivm/casus-datasets/*.csv"
  gzip covid-19-master/data-rivm/casus-datasets/*.csv
  mv covid-19-master/data-rivm/casus-datasets/*.csv .
  gzip -v *.csv

  # delete empty directories
  rm -ri covid-19-master/

In Windows, you can download master.zip from the web browser and do
the other steps from a Git bash shell.
