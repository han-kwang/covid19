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

Here are the Python files (list is out of date):

  casus_analysis.py - module for parsing/analyzing NL casus data.
  chen_airborne_exposure_plot.py - analysis of data on droplet range.
  holiday_regions.py - mapping municipalities to school-holiday regions
  nl_regions.py - module with functions for handling municipalities/regions.
  nlcovidstats.py - module with functions to process the positives.
  nlcovidstats_show.py - script to generate plots.
  plot_R_from_daily_cases.py - another script, just plot R number.
  process_casus_data.py - script demonstrating casus analysis.
  run_seirmodel.py - script demonstrating the SEIR model.
  seirmodel.py - SEIR epidimiological model (not maintained).
  test_seirmodel.py - test cases for SEIR model
  tools.py - helper functions.

To install pandas in Linux:
  pip3 install pandas scipy matplotlib

If you encounter "locale.Error: unsupported locale setting", solve
this by adding nl_NL.UTF-8 in:

  sudo dpkg reconfigure-locales

In order to use casus_analysis/process_casus_data (not for plotting
R values over time), you need to download the archive of casus data.
In Linux:

  cd data-casus
  wget https://github.com/mzelst/covid-19/archive/master.zip
  unzip master.zip "covid-19-master/data-rivm/casus-datasets/*.csv"
  mv covid-19-master/data-rivm/casus-datasets/*.csv .
  gzip -v *.csv
  rm -i master.zip

  # delete empty directories
  rm -ri covid-19-master/

In Windows, you can download master.zip from the web browser and do
the other steps from a Git bash shell.

If you have questions, you can contact me via Twitter: @hk_nien.
