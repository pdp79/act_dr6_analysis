#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################################################
# The script will take care of initializing the directory structure assumed
# throught the package and downloading/generating the data/priors to use. 
#
# @author: Patricia Diego-Palazuelos (diegop@mpa-garching.mpg.de)
#####################################################################

import os, stat

path = os.path.dirname(os.path.realpath(__file__))
print(f'Working from {path}')
# change file permissions to allow execution
for file in ['binning', 'chi_square', 'cov', 'linear_system', 'mle', 'priors']:
    os.chmod(f'{path}/{file}.py', stat.S_IRUSR|stat.S_IWUSR|stat.S_IXUSR)
    
    
# create results directory if it does not exist already
res_dir = f'{path}/results'
if not os.path.isdir(res_dir):
    print('Creating results directory')
    os.mkdir(res_dir)

# create priors if they don't exist already
prior_dir = f'{path}/priors'
if not os.path.isdir(prior_dir):
    print('Creating priors directory')
    os.mkdir(prior_dir)
    os.system(f'{path}/priors.py {prior_dir}')
    
# download data if it wasn't done before
data_dir = f'{path}/dr6_data'
bin_dir  = f'{data_dir}/binning'
sacc_dir = f'{data_dir}/sacc_files' 
lcdm_dir = f'{data_dir}/best_fits'
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    os.mkdir(bin_dir)
    os.mkdir(lcdm_dir)
    os.mkdir(sacc_dir)
    
    print('Downloading ACT DR6 data')
    url  = 'https://lambda.gsfc.nasa.gov/'
    
    # download
    os.system(f'wget -q -t 5 -nc -w 3 {url}data/act/pspipe/best_fits/act_dr6.02_best_fits_dr6_lcdm.tar.gz -O {lcdm_dir}/best_fits_dr6_lcdm.tar.gz')
    os.system(f'wget -q -t 5 -nc -w 3 {url}data/act/pspipe/sacc_files/dr6_data.tar.gz -O {sacc_dir}/dr6_data.tar.gz')
    for nb in ['50', '20']:
        os.system(f'wget -q -t 5 -nc -w 3 {url}data/act/pspipe/binning/binning_{nb} -O {bin_dir}/binning_{nb}')

    # unzip
    os.system(f'tar -zxvf {lcdm_dir}/best_fits_dr6_lcdm.tar.gz -C {lcdm_dir}/ dr6_lcdm_best_fits/cmb.dat')
    os.system(f'tar -zxvf {sacc_dir}/dr6_data.tar.gz -C {sacc_dir}/ v1.0/dr6_data.fits')
    # clean up
    os.system(f'rm -r {lcdm_dir}/best_fits_dr6_lcdm.tar.gz')
    os.system(f'rm -r {sacc_dir}/dr6_data.tar.gz')
