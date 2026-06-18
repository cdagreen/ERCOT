# ERCOT

Analysis of the seasonal pattern of hourly electricity demand in the ERCOT (Electric Reliability Council of Texas) region. 

Includes exploratory analysis of annual, weekly, and intraday seasonality, as well as smoothing spline modeling tools. 

## Data Retrieval

**eia_ercot_data.py** provides tools for retrieving data from the EIA API (requires an API key) https://www.eia.gov/opendata/

**retrieve_ercot_data.py** uses these tools to constuct the analysis dataset

## Exploratory Analysis

**exploratory.py** provides tools for exploratory analysis (data loading, plotting, etc.)

**exploraroty.ipynb** contains the main exploratory analysis. 

## Modeling

**cyclic.py** provides tools for modeling seasonality using smoothing splines, as well as seasonal-trend decomposition

**seasonality_models.py** applies some of the tools from cyclic.py (in progress)


