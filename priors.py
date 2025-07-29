#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################################################
# Define priors to use. 
# They will be identified by the names given in this script.
#
# Array-bands are ordered like
# bands = ['pa4_f220', 'pa5_f090', 'pa5_f150', 'pa6_f090', 'pa6_f150']
#
# @author: Patricia Diego-Palazuelos (diegop@mpa-garching.mpg.de)
#####################################################################
import numpy as np
import pickle
import argparse, textwrap

def initial():
    """
    Function to read input arguments from command line and to offer help to user.
    """
    parser = argparse.ArgumentParser(
                    prog='priors.py',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    description=textwrap.dedent('''
                            Script generating the priors to use in the analysis
                            '''))
    parser.add_argument('dir', metavar='dir', type=str, nargs=1,
                        help='absolute path to prior directory')
    args      = parser.parse_args()
    return args

args = initial()
path = args.dir[0] 


#######################################################################
### systematic uncertainty from optical modeling, different for each PA
### Both including/excluding PA4
### No correlation between angles/PA/frequencies
#######################################################################
tag_1 = 'diag0.09-0.09-0.11'

prior_vec_1       = np.zeros(5)
prior_cov_1       = np.zeros((5, 5))
prior_cov_1[0, 0] = np.deg2rad(0.09)**2
prior_cov_1[1, 1] = prior_cov_1[2, 2] = np.deg2rad(0.09)**2
prior_cov_1[3, 3] = prior_cov_1[4, 4] = np.deg2rad(0.11)**2

with open(f"{path}/{tag_1}_pa56.pkl" , 'wb') as handle:
    pickle.dump({'vec': prior_vec_1[1:], 'cov': prior_cov_1[1:, 1:]}, 
                handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f"{path}/{tag_1}_pa456.pkl" , 'wb') as handle:
     pickle.dump({'vec': prior_vec_1, 'cov': prior_cov_1}, 
                 handle, protocol=pickle.HIGHEST_PROTOCOL)


#######################################################################
### systematic uncertainty from optical modeling, different for each PA
### Both including/excluding PA4
### 90% correlation within the same PA
#######################################################################
rho_2 = 90
tag_2 = f'diag0.09-0.09-0.11_PA{rho_2}'

prior_vec_2 = np.zeros(5)
prior_cov_2 = np.zeros((5, 5))
s_pa4_2     = np.deg2rad(0.09)
s_pa5_2     = np.deg2rad(0.09)
s_pa6_2     = np.deg2rad(0.11)

prior_cov_2[0, 0] = s_pa4_2**2

prior_cov_2[1, 1] = prior_cov_2[2, 2] = s_pa5_2**2
prior_cov_2[1, 2] = prior_cov_2[2, 1] = (rho_2/100)*s_pa5_2**2

prior_cov_2[3, 3] = prior_cov_2[4, 4] = s_pa6_2**2
prior_cov_2[3, 4] = prior_cov_2[4, 3] = (rho_2/100)*s_pa6_2**2

with open(f"{path}/{tag_2}_pa56.pkl" , 'wb') as handle:
    pickle.dump({'vec': prior_vec_2[1:], 'cov': prior_cov_2[1:, 1:]},
                handle, protocol=pickle.HIGHEST_PROTOCOL)
     
with open(f"{path}/{tag_2}_pa456.pkl" , 'wb') as handle:
     pickle.dump({'vec': prior_vec_2, 'cov': prior_cov_2},
                 handle, protocol=pickle.HIGHEST_PROTOCOL)
     

#######################################################################
### systematic uncertainty from optical modeling, different for each PA
### Both including/excluding PA4
### 90% correlation within the same frequency
#######################################################################
rho_3 = 90
tag_3 = f'diag0.09-0.09-0.11_nu{rho_3}' 

prior_vec_3 = np.zeros(5)
prior_cov_3 = np.zeros((5, 5))
s_pa4_3     = np.deg2rad(0.09)
s_pa5_3     = np.deg2rad(0.09)
s_pa6_3     = np.deg2rad(0.11)


prior_cov_3[0, 0] = s_pa4_3**2

prior_cov_3[1, 1] = prior_cov_3[2, 2] = s_pa5_3**2

prior_cov_3[3, 3] = prior_cov_3[4, 4] = s_pa6_3**2

prior_cov_3[1, 3] = prior_cov_3[3, 1] = (rho_3/100) *s_pa5_3*s_pa6_3
prior_cov_3[2, 4] = prior_cov_3[4, 2] = (rho_3/100) *s_pa5_3*s_pa6_3


with open(f"{path}/{tag_3}_pa56.pkl" , 'wb') as handle:
    pickle.dump({'vec': prior_vec_3[1:], 'cov': prior_cov_3[1:, 1:]},
                handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f"{path}/{tag_3}_pa456.pkl" , 'wb') as handle:
     pickle.dump({'vec': prior_vec_3, 'cov': prior_cov_3},
                 handle, protocol=pickle.HIGHEST_PROTOCOL)
 


