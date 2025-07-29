#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################################################
# Functions to emulate NaMaster binning without using NaMaster.
# This version is compatible with ACT binning
#
# @author: Patricia Diego-Palazuelos (diegop@mpa-garching.mpg.de)
#####################################################################

import numpy as np 
import warnings
from act_dr6_analysis import bin_dir


def indices(m1, f1, m2, f2):
    """
    Retrieve a particular entry in ACT DR6 data vector/matrices

    Parameters
    ----------
    m1 : str
        Variable (T,E,B) number 1
    f1 : TYPE
        Cross array-band number 1
    m2 : str
        Variable (T,E,B) number 2
    f2 : TYPE
        Cross array-band number 2

    Returns
    -------
    int
        Starting index such that array[start_idx:end_idx] returns the desired information
    int
        Ending index such that array[start_idx:end_idx] returns the desired information

    """
    nb  = 57 # all the bins stored in DR6
    idx = sacc_order[f'{m1}{m2}'][f'{m1} {f1} x {m2} {f2}']
    return idx*nb, (idx+1)*nb
    

# Order in which variables are hard coded within ACT sacc files
sacc_order = {
                'TT':{'T pa4_f220 x T pa4_f220':0,
                      'T pa4_f220 x T pa5_f090':1, 'T pa5_f090 x T pa4_f220':1,
                      'T pa4_f220 x T pa5_f150':2, 'T pa5_f150 x T pa4_f220':2,
                      'T pa4_f220 x T pa6_f090':3, 'T pa6_f090 x T pa4_f220':3,
                      'T pa4_f220 x T pa6_f150':4, 'T pa6_f150 x T pa4_f220':4,
                      #########################################################
                      'T pa5_f090 x T pa5_f090':5, 
                      'T pa5_f090 x T pa5_f150':6, 'T pa5_f150 x T pa5_f090':6, 
                      'T pa5_f090 x T pa6_f090':7, 'T pa6_f090 x T pa5_f090':7,
                      'T pa5_f090 x T pa6_f150':8, 'T pa6_f150 x T pa5_f090':8,           
                      #########################################################
                      'T pa5_f150 x T pa5_f150':9, 
                      'T pa5_f150 x T pa6_f090':10, 'T pa6_f090 x T pa5_f150':10,
                      'T pa5_f150 x T pa6_f150':11, 'T pa6_f150 x T pa5_f150':11, 
                      #########################################################            
                      'T pa6_f090 x T pa6_f090':12, 
                      'T pa6_f090 x T pa6_f150':13, 'T pa6_f150 x T pa6_f090':13,
                      #########################################################
                      'T pa6_f150 x T pa6_f150':14},
              
                'TE':{'T pa4_f220 x E pa4_f220':15,
                      'T pa4_f220 x E pa5_f090':16, 'T pa5_f090 x E pa4_f220':20,
                      'T pa4_f220 x E pa5_f150':17, 'T pa5_f150 x E pa4_f220':21,
                      'T pa4_f220 x E pa6_f090':18, 'T pa6_f090 x E pa4_f220':22,
                      'T pa4_f220 x E pa6_f150':19, 'T pa6_f150 x E pa4_f220':23,
                      #########################################################
                      'T pa5_f090 x E pa5_f090':24, 
                      'T pa5_f090 x E pa5_f150':25, 'T pa5_f150 x E pa5_f090':28, 
                      'T pa5_f090 x E pa6_f090':26, 'T pa6_f090 x E pa5_f090':29,
                      'T pa5_f090 x E pa6_f150':27, 'T pa6_f150 x E pa5_f090':30,           
                      #########################################################
                      'T pa5_f150 x E pa5_f150':31, 
                      'T pa5_f150 x E pa6_f090':32, 'T pa6_f090 x E pa5_f150':34,
                      'T pa5_f150 x E pa6_f150':33, 'T pa6_f150 x E pa5_f150':35, 
                      #########################################################            
                      'T pa6_f090 x E pa6_f090':36, 
                      'T pa6_f090 x E pa6_f150':37, 'T pa6_f150 x E pa6_f090':38,
                      #########################################################
                      'T pa6_f150 x E pa6_f150':39},
              
                'TB':{'T pa4_f220 x B pa4_f220':40,
                      'T pa4_f220 x B pa5_f090':41, 'T pa5_f090 x B pa4_f220':45,
                      'T pa4_f220 x B pa5_f150':42, 'T pa5_f150 x B pa4_f220':46,
                      'T pa4_f220 x B pa6_f090':43, 'T pa6_f090 x B pa4_f220':47,
                      'T pa4_f220 x B pa6_f150':44, 'T pa6_f150 x B pa4_f220':48,
                      #########################################################
                      'T pa5_f090 x B pa5_f090':49, 
                      'T pa5_f090 x B pa5_f150':50, 'T pa5_f150 x B pa5_f090':53, 
                      'T pa5_f090 x B pa6_f090':51, 'T pa6_f090 x B pa5_f090':54,
                      'T pa5_f090 x B pa6_f150':52, 'T pa6_f150 x B pa5_f090':55,           
                      #########################################################
                      'T pa5_f150 x B pa5_f150':56, 
                      'T pa5_f150 x B pa6_f090':57, 'T pa6_f090 x B pa5_f150':59,
                      'T pa5_f150 x B pa6_f150':58, 'T pa6_f150 x B pa5_f150':60, 
                      #########################################################            
                      'T pa6_f090 x B pa6_f090':61, 
                      'T pa6_f090 x B pa6_f150':62, 'T pa6_f150 x B pa6_f090':63,
                      #########################################################
                      'T pa6_f150 x B pa6_f150':64},
              
                'EE':{'E pa4_f220 x E pa4_f220':65,
                      'E pa4_f220 x E pa5_f090':66, 'E pa5_f090 x E pa4_f220':66,
                      'E pa4_f220 x E pa5_f150':67, 'E pa5_f150 x E pa4_f220':67,
                      'E pa4_f220 x E pa6_f090':68, 'E pa6_f090 x E pa4_f220':68,
                      'E pa4_f220 x E pa6_f150':69, 'E pa6_f150 x E pa4_f220':69,
                      #########################################################
                      'E pa5_f090 x E pa5_f090':70, 
                      'E pa5_f090 x E pa5_f150':71, 'E pa5_f150 x E pa5_f090':71, 
                      'E pa5_f090 x E pa6_f090':72, 'E pa6_f090 x E pa5_f090':72,
                      'E pa5_f090 x E pa6_f150':73, 'E pa6_f150 x E pa5_f090':73,           
                      #########################################################
                      'E pa5_f150 x E pa5_f150':74, 
                      'E pa5_f150 x E pa6_f090':75, 'E pa6_f090 x E pa5_f150':75,
                      'E pa5_f150 x E pa6_f150':76, 'E pa6_f150 x E pa5_f150':76, 
                       #########################################################            
                       'E pa6_f090 x E pa6_f090':77, 
                       'E pa6_f090 x E pa6_f150':78, 'E pa6_f150 x E pa6_f090':78,
                       #########################################################
                       'E pa6_f150 x E pa6_f150':79},
              
                'EB':{'E pa4_f220 x B pa4_f220':80,
                      'E pa4_f220 x B pa5_f090':81, 'E pa5_f090 x B pa4_f220':95,
                      'E pa4_f220 x B pa5_f150':82, 'E pa5_f150 x B pa4_f220':96,
                      'E pa4_f220 x B pa6_f090':83, 'E pa6_f090 x B pa4_f220':97,
                      'E pa4_f220 x B pa6_f150':84, 'E pa6_f150 x B pa4_f220':98,
                      #########################################################
                      'E pa5_f090 x B pa5_f090':85, 
                      'E pa5_f090 x B pa5_f150':86, 'E pa5_f150 x B pa5_f090':99, 
                      'E pa5_f090 x B pa6_f090':87, 'E pa6_f090 x B pa5_f090':100,
                      'E pa5_f090 x B pa6_f150':88, 'E pa6_f150 x B pa5_f090':101,           
                      #########################################################
                      'E pa5_f150 x B pa5_f150':89, 
                      'E pa5_f150 x B pa6_f090':90, 'E pa6_f090 x B pa5_f150':102,
                      'E pa5_f150 x B pa6_f150':91, 'E pa6_f150 x B pa5_f150':103, 
                      #########################################################            
                      'E pa6_f090 x B pa6_f090':92, 
                      'E pa6_f090 x B pa6_f150':93, 'E pa6_f150 x B pa6_f090':104,
                      #########################################################
                      'E pa6_f150 x B pa6_f150':94},
                
                'BB':{'B pa4_f220 x B pa4_f220':105,
                      'B pa4_f220 x B pa5_f090':106, 'B pa5_f090 x B pa4_f220':106,
                      'B pa4_f220 x B pa5_f150':107, 'B pa5_f150 x B pa4_f220':107,
                      'B pa4_f220 x B pa6_f090':108, 'B pa6_f090 x B pa4_f220':108,
                      'B pa4_f220 x B pa6_f150':109, 'B pa6_f150 x B pa4_f220':109,
                      #########################################################
                      'B pa5_f090 x B pa5_f090':110, 
                      'B pa5_f090 x B pa5_f150':111, 'B pa5_f150 x B pa5_f090':111, 
                      'B pa5_f090 x B pa6_f090':112, 'B pa6_f090 x B pa5_f090':112,
                      'B pa5_f090 x B pa6_f150':113, 'B pa6_f150 x B pa5_f090':113,           
                      #########################################################
                      'B pa5_f150 x B pa5_f150':114, 
                      'B pa5_f150 x B pa6_f090':115, 'B pa6_f090 x B pa5_f150':115,
                      'B pa5_f150 x B pa6_f150':116, 'B pa6_f150 x B pa5_f150':116, 
                      #########################################################            
                      'B pa6_f090 x B pa6_f090':117, 
                      'B pa6_f090 x B pa6_f150':118, 'B pa6_f150 x B pa6_f090':118,
                      #########################################################
                      'B pa6_f150 x B pa6_f150':119}
             }



class BandPowers:
    
    def __init__(self, opt, lmin=None, lmax=None, nlb='50'):
        """
        Object containing the details of the DR6 band power definition.

        Parameters
        ----------
        opt : str
            Choose between 'fixed', 'baseline', 'extended'. 
            'fixed' applies a common lmin-lmax range to all array-bands.
            'baseline' and 'extended' correspond to the multipole ranges defined
            by ACT DR6 [arXiv:2503.14452].
        lmin : int, optional unless opt=='fixed'
            Center of the minimum multipole bin to include. The default is None.
        lmax : int, optional unless opt=='fixed'
            Center of the maximum multipole to include. The default is None.
        nlb : str, optional
            Typical number of multipoles per bin in ACT DR6 products. 
            The default is '50' but '20' is also allowed.

        Raises
        ------
        NotImplementedError
            If opt is not one of 'fixed', 'baseline', 'extended'.
        ValueError
            If opt=='fixed' but no lmin, lmax were specified.
            If nlb different from the ['20', '50'] supported by ACT DR6

        Returns
        -------
        None.

        """
        if opt not in ['fixed', 'baseline', 'extended']:
            raise NotImplementedError(f"'{opt}' is not a valid binning configuration")
        if opt=='fixed' and (lmin==None or lmax==None):
            raise ValueError(f"When choosing '{opt}', lmin and lmax must be specified")
        if nlb == '20':
            warnings.warn(f"nlb={nlb} allowed but not tested. Proceed at your discretion.")
        elif nlb not in ['20', '50']:
            raise ValueError(f"nlb={nlb} not supported by ACT DR6.")
            
        self.opt    = opt
        self.nlb    = nlb
        bin_file    = np.loadtxt(f'{bin_dir}/binning_{self.nlb}')
        
        # Values to save at first, generic just to cover the full range
        if opt=='fixed':
            # In this case, they will be the definitive ones
            self.g_lmin = lmin
            self.g_lmax = lmax
        else:
            self.g_lmin = 0
            self.g_lmax = 8000
        g_range     = (bin_file[:,2] >=self.g_lmin) & (bin_file[:,2] < self.g_lmax)
        self.g_b    = bin_file[g_range, 2]
        self.g_Nb   = len(self.g_b)
        self.g_info = binning_from_edges(bin_file[g_range, 0].astype(int), 
                                         bin_file[g_range, 1].astype(int))
        # Save to a dictionary
        self.conf   = {'pa4_f220':None, 
                       'pa5_f090':None, 'pa5_f150':None, 
                       'pa6_f090':None, 'pa6_f150':None}
        
        if self.opt == 'fixed':
            # Common lmin-lmax for all bands
            com_conf = get_conf(lmin, lmax, bin_file)
            self.conf['pa4_f220'] = com_conf
            self.conf['pa5_f090'] = com_conf
            self.conf['pa5_f150'] = com_conf
            self.conf['pa6_f090'] = com_conf
            self.conf['pa6_f150'] = com_conf
            
        elif self.opt == 'baseline':
            # Definition given by ACT DR6 [arXiv:2503.14452]
            # Paper says 8500 but sacc does not include that bin
            # I fixed the range for pa4
            self.conf['pa4_f220'] = get_conf(1000, 8000, bin_file)
            self.conf['pa5_f090'] = get_conf(1000, 8000, bin_file)
            self.conf['pa5_f150'] = get_conf(800,  8000, bin_file)
            self.conf['pa6_f090'] = get_conf(1000, 8000, bin_file)
            self.conf['pa6_f150'] = get_conf(600,  8000, bin_file) 
            
        elif self.opt == 'extended':
            # Definition given by ACT DR6 [arXiv:2503.14452]
            # Paper says 8500 but sacc does not include that bin
            # I fixed the range for pa4
            com_conf = get_conf(500, 8000, bin_file)
            self.conf['pa4_f220'] = com_conf
            self.conf['pa5_f090'] = com_conf
            self.conf['pa5_f150'] = com_conf
            self.conf['pa6_f090'] = com_conf
            self.conf['pa6_f150'] = com_conf
                    


def get_conf(s, e, file):
    """
    Return details on the ACT DR6 binning configuration.

    Parameters
    ----------
    s : int
        Initial multipole to include.
    e : int
        Final multipole to include.
    file : np.array
        Array especifying the ACT DR6 binning definition.

    Returns
    -------
    dict
        Dictionary with all the binning-related information needed later on.

    """
    b_range  = (file[:,2] >= s) & (file[:,2] < e)
    b_center = file[b_range, 2]
    Nb       = len(b_center)
    binInfo  = binning_from_edges(file[b_range, 0].astype(int), 
                                  file[b_range, 1].astype(int))
    return {'lmin':s, 'lmax':e, 'Nbins':Nb, 'b':b_center, 'info':binInfo}
        


def binning_from_edges(ell_ini, ell_end):
    """
    Calculate the configuration of multipoles per bin assuming uniform binning.

    Parameters
    ----------
    ell_ini : int
        Initial multipole to include.
    ell_end : int
        Final multipole to include.

    Returns
    -------
    n_bands : int
        Number of bins.
    nell_array : np.array
        Number of multipoles per bin.
    ell_list : list
        Multipoles in each bin.
    w_list : list
        Weight that each multipole has in every bin.

    """
    # Reproducing NaMaster's functions
    # First function
    nls  = np.amax(ell_end)
    lmax = nls
    ells, bpws, weights = [], [], []
    for ib, (li, le) in enumerate(zip(ell_ini, ell_end)):
        # To reproduce ACT binning
        nlb      = int(le - li) + 1
        ells    += list(range(li, le + 1)) # Include the li and le
        bpws    += [ib] * nlb
        weights += [1./nlb] * nlb # Uniform weights, just the mean
    ells = np.array(ells); bpws = np.array(bpws); weights = np.array(weights)

    # Second function
    nell    = len(ells)
    n_bands = bpws[-1]+1
    
    nell_array = np.zeros(n_bands, dtype=int)
    for ii in range(0, nell, 1):
        if ells[ii]<=lmax and bpws[ii]>=0:
            nell_array[bpws[ii]]+=1   
            
    ell_list=[]; w_list=[]
    for ii in range(0, n_bands, 1):
        ell_list.append([])
        w_list.append([])
    
    for ii in range(0, nell, 1):
        if ells[ii]<=lmax and bpws[ii]>=0: 
            ell_list[bpws[ii]].append(ells[ii]) 
            w_list[bpws[ii]].append(weights[ii])

    return (n_bands, nell_array, ell_list, w_list)


def bin_cls(spec, info):
    """
    Apply uniform binning to an input angular power spectrum.

    Parameters
    ----------
    spec : np.array
        Array with the input spectrum to bin.
    info : tuple with the output of binning_from_edges()
        Configuration of the uniform binning to apply.

    Returns
    -------
    spec_out : np.array
        Array with the output binned spectrum.

    """
    # Unpack
    (n_bands, nell_array, ell_list, w_list) = info
    spec_out = np.zeros(n_bands, dtype=np.float64)
    for ii in range(n_bands):
        # Assuming cl start from ell=0
        spec_out[ii] = np.sum( spec[ell_list[ii]] * w_list[ii] )
    return spec_out
