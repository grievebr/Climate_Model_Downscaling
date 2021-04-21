# This code provides all the steps required to take raw, low-resolution IPCC CMIP5 ocean models,
# debias and downscale them to a high-resolution Regional Ocean Modeling System (ROMS) ocean
# model, and turn them into an ensemble. This will result in high resolution (~7km) debiased projections
# of the requested ocean variables in the 2080-2100 period under different climate scenarios.
#
#
# With the ensemble_env function, users can select the following based on what they need the data for: 
#
# Variables:
# potential temperature ('thetao')
# potential salinity ('so')
#
# Experiment:
# RCP 8.5 ('rcp85'), a high emissions climate scenario 
# RCP 4.5 ('rcp45'), a medium emissions climate scenario
#
# Depth:
# Surface water ('surface')
# Bottom water ('bottom')


import numpy as np
import xarray as xr
import cftime
import warnings
from scipy.interpolate import griddata
from scipy.io import loadmat

# Load existing ROMS climatology to be used as the basis for downscaling and debiasing
# These are stored as Matlab .mat files, but are easily converted
romdict = loadmat('C:/Users/griev/Documents/OceanModels/roms.mat')
rom_sst = romdict['sst']; # Sea Surface Temperature
rom_sss = romdict['sss']; # Sea Surface Salinity
rom_bwt = romdict['bwt']; # Bottom Water Temperature
rom_bws = romdict['bws']; # Bottom Water Salinity
rom_bath = romdict['bath_r']; # ROMS Bathymetry
rom_lon = romdict['lon_r']; # Curvelinear ROMS longitude matrix
rom_lat = romdict['lat_r'];# Curvelinear ROMS latitude matrix


# Modeling centers included in ensemble
modelnames = ['CanESM','CMCC','MIROC']

# Python produces a warning every time arrays are averaged containing NaN at the same point in all arrays
# Because land is treated as NaN, this warning occurs on every map. Run command to suppress
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

#%%
# The following three functions will be used within ensemble_var 

#%%
# Function to find deepest informative layer at each point of a Z-layered climate model
# Input must be a 4D array with the dimensions [Months Levels Lat Lon]
# Code goes through each layer in water column and overrides saved key if layer is informative

def bottom_layer_fun(array_in):
    m,levels,j,k = array_in.shape
    bottomlayer = np.zeros([j,k]);
    for lev in range(levels):
        layer = array_in[0,lev,:,:] # Information constant over time, so this doesn't need repeated
        fin = np.where(np.isfinite(layer))
        bottomlayer[fin] = lev;    
    return bottomlayer
        

#%%
# Function to find the average monthly bottom water variable of a climate model
# Will average over many years to reduce interannual variability
# Input must be a 4D array with the dimensions [Months Levels Lat Lon]
# along with key from bottom_layer_fun. Vectorizes and reshapes maps to reduce looping

def model_bottom_avg(array_in, bottomlayer_in):
    m,levels,j,k = array_in.shape   
    flatbottom = bottomlayer_in.flatten() # Vectorize
    bot_var = np.empty([12,j,k])
    
    for month in range(12):
        i_mon = range(month,m,12) # Same months, different years
        y_num = len(i_mon)
        year_maps = np.empty([y_num,j,k])
        for year in range(y_num):
            mapslice = np.array(array_in[i_mon[year],:,:,:]).reshape([levels,j*k]) # all points and levels to be keyed with flatbottom
            bottom_vec = np.zeros([j*k,])
            for ind in range(len(bottom_vec)):
                bottom_vec[ind] = mapslice[int(flatbottom[ind]),ind] # pull value at deepest layer
            year_maps[year,:,:] = bottom_vec.reshape(j,k)
        bot_var[month,:,:] = np.nanmean(year_maps, axis=0)
        
    return bot_var



#%%
# Function to find the average monthly surface water variable of a climate model
# Will average over many years to reduce interannual variability
# Input must be a 4D array with the dimensions [Months Levels Lat Lon]
# Faster than finding bottom variables

def model_surface_avg(array_in):
    m,levels,j,k = array_in.shape   
    surf_var = np.empty([12,j,k])
    for month in range(12):
        i_mon = range(month,m,12)
        ssv_mon = array_in[i_mon,0,:,:]; 
        surf_var[month,:,:] = np.nanmean(ssv_mon, axis=0)
        
    return surf_var


#%%
# Primary function that will pull saved IPCC model data for both future ('scen') and historical ('clim') periods,
# call prior functions to create 20-year averages, and downscale them via the delta method.
# Output will be an averaged ensemble of size [12 x y] 
# Inputs are strings of the desired variable, scenario, and depth as described at the start of the script
#  

def ensemble_var(var, scenario, depth):
    gcms = []; # store each member of ensemble
    for mod in modelnames:
        # Load climatology
        clim_filename = 'C:/Users/griev/Documents/OceanModels/%s/%s/%s/*.nc' % (mod, var, 'historical');
        clim_nc = xr.open_mfdataset(clim_filename); #Multifile dataset eliminates need to concat later 

        # load RCP scenario
        scen_filename = 'C:/Users/griev/Documents/OceanModels/%s/%s/%s/*.nc' % (mod, var, scenario);
        scen_nc = xr.open_mfdataset(scen_filename);
        lat = scen_nc.lat;
        lon = scen_nc.lon;

        # Require 2D gridded latitude and longitude
        if len(lon.dims)==1:
            lon2, lat2 = np.meshgrid(lon,lat, sparse=False, indexing='xy')
        elif len(lon.dims)==2:
            lon2 = np.array(lon);
            lat2 = np.array(lat);
        
        # Retrieve desired variables and create long-term averages using previous functions
        if var=='thetao' and depth=='surface':
            clim_var_array = clim_nc.thetao - 273.15; # stored in Kelvin
            scen_var_array = scen_nc.thetao - 273.15;
            rom_array = rom_sst;
            var_clim = model_surface_avg(clim_var_array)
            var_scen = model_surface_avg(scen_var_array)
        
        if var=='thetao' and depth=='bottom':
            clim_var_array = clim_nc.thetao - 273.15; 
            scen_var_array = scen_nc.thetao - 273.15;
            rom_array = rom_bwt;
            bottomlayer = bottom_layer_fun(clim_var_array);
            var_clim = model_bottom_avg(clim_var_array, bottomlayer)
            var_scen = model_bottom_avg(scen_var_array, bottomlayer)
            
        if var=='so' and depth=='surface':
            clim_var_array = clim_nc.so;
            scen_var_array = scen_nc.so;
            rom_array = rom_sss;
            var_clim = model_surface_avg(clim_var_array)
            var_scen = model_surface_avg(scen_var_array)
        
        if var=='so' and depth=='bottom':
            clim_var_array = clim_nc.so; 
            scen_var_array = scen_nc.so;
            rom_array = rom_bws;
            bottomlayer = bottom_layer_fun(clim_var_array);
            var_clim = model_bottom_avg(clim_var_array, bottomlayer)
            var_scen = model_bottom_avg(scen_var_array, bottomlayer)
        
       
        # close .nc files
        clim_nc.close();
        scen_nc.close();
        
        # regrid to ROMS resolution using nearest neighbor interpolation
        var_clim_regrid = np.empty([12,*rom_lat.shape])
        var_clim_regrid[:] = np.nan;
        var_scen_regrid = np.empty([12,*rom_lat.shape])
        var_scen_regrid[:] = np.nan;
        for month in range(12):
            var_clim_regrid[month] = griddata((lon2.flatten(), lat2.flatten()), var_clim[month].flatten(), (rom_lon,rom_lat), method='nearest')
            var_scen_regrid[month] = griddata((lon2.flatten(), lat2.flatten()), var_scen[month].flatten(), (rom_lon,rom_lat), method='nearest')
            
        # Take change in variable with future climate scenario (delta) and add that to baseline ROMS model    
        delta = var_scen_regrid - var_clim_regrid;
        downscaled_gcm = rom_array + delta;
        
        gcms.append(downscaled_gcm)
        
    return np.nanmean(gcms,axis=0)
            
#%%        
# Examples
bwt85 = ensemble_var('thetao','rcp85', 'bottom')
sss45 = ensemble_var('so','rcp45','surface')

