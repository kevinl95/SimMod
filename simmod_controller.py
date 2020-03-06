import numpy as np

from constants import *
from emissions_parser import emissions
from concs_pulse_decay import pulse_decay_runner
from radiative_forcing import calc_radiative_forcing
from heat_diffusion import continuous_diffusion_model

#Model Parameters
run_start_year = 1765.          #Run start year
run_end_year = 2100.            #Inclusive of end year
dt = 1 #/ 100.                  #years
rcp = '8.5'                     #RCP scenario
carbon_model = 'pulse response' #'pulse response' or 'box diffusion'
normalize_2000_conc = True      #Normalize concentrations to historical year-2000 values
c_sens = 1.25                   #Climate sensativity (T = F / LAMBDA)

#Ocean Box Diffusion Model Settings (when relevant)
MIXING = 'probable'             #options 'fast', 'slow', or 'probable'
DZ = 100                        #meters - thickness of each layer in the deep ocean


def run_simmod(run_start_year, run_end_year, dt, rcp, c_sens = c_sens, add_start = 0,
               add_end = 0, c_add = 0, ch4_add = 0, n2o_add = 0):
    """
    Run the various parts of SimMod and export images and CSV files.
    """
    run_years = (run_end_year - run_start_year + 1)
    emission_vals = emissions(run_start_year, run_end_year, dt, rcp,
                              add_start, add_end, c_add, ch4_add, n2o_add)
    conc = pulse_decay_runner(run_years, dt, emission_vals)

    if carbon_model == 'box diffusion':
        box_diffusion_results = box_diffusion_model(
            emission_vals,
            dt,
            DZ,
            MIXING
        )
        conc['co2_ppm'] = box_diffusion_results['co2ppm']

    if normalize_2000_conc == True:
        conc['co2_ppm'] = (
            conc['co2_ppm'] -
            conc.loc[conc['year'] == 2000, 'co2_ppm'].min() +
            emission_vals.loc[emission_vals['year'] == 2000, 'rcp_co2_ppm'].min()
        )
        conc['ch4_ppb'] = (
            conc['ch4_ppb'] -
            conc.loc[conc['year'] == 2000, 'ch4_ppb'].min() +
            emission_vals.loc[emission_vals['year'] == 2000, 'rcp_ch4_ppb'].min()
        )
        conc['n2o_ppb'] = (
            conc['n2o_ppb'] -
            conc.loc[conc['year'] == 2000, 'n2o_ppb'].min() +
            emission_vals.loc[emission_vals['year'] == 2000, 'rcp_n2o_ppb'].min()
        )

    forcing = calc_radiative_forcing(conc)
    warming = continuous_diffusion_model(forcing, run_years, dt, c_sens)
    return warming

results = run_simmod(run_start_year, run_end_year, dt, rcp, c_sens)
results.to_csv('results/simmod_run_'+rcp+' '+carbon_model+'.csv')
