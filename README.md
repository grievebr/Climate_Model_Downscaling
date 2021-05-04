This code provides all the steps required to take raw, low-resolution IPCC CMIP5 ocean models,
debias and downscale them to a high-resolution Regional Ocean Modeling System (ROMS) ocean
model, and turn them into an ensemble. This will result in high resolution (~7km) debiased projections
of the requested ocean variables in the 2080-2100 period under different climate scenarios.


With the ensemble_env function, users can select the following based on what they need the data for: 

Variables:
potential temperature ('thetao')
potential salinity ('so')

Experiment:
RCP 8.5 ('rcp85'), a high emissions climate scenario 
RCP 4.5 ('rcp45'), a medium emissions climate scenario

Depth:
Surface water ('surface')
Bottom water ('bottom')
