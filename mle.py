#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################################################
# Minami-Komatsu fit to
# EB_obs = rot(alpha)*(EE_obs - BB_obs) + rot(alpha, beta)*(EE_cmb - BB_cmb)
#          + calibration priors on alpha
# TB_obs = rot(alpha)*TE_obs + rot(alpha, beta)*TE_cmb
#          + calibration priors on alpha       
# 
# * Ignore foreground contributions (neglegible at ACT scales)
# * Use the EE, BB, EB obs angular power spectra provided by ACT
# * The covariance is already corrected for partial sky coverage,
# * Power spectra are already deconvolved by beam and pixel window functions
# * For variable binning, keep the most restrictive of the two bands 
#
# @author: Patricia Diego-Palazuelos (diegop@mpa-garching.mpg.de) 
#####################################################################
import numpy as np
from scipy.stats import chi2
import os, pickle, sacc, warnings
from act_dr6_analysis import res_dir, prior_dir, sacc_dir, lcdm_dir
import act_dr6_analysis.binning as bf
from act_dr6_analysis.cov import MK_cov
from act_dr6_analysis.linear_system import MK_linear_system
from act_dr6_analysis.chi_square import MK_chi_square

# common parameters
max_iter   = 100
rad2arcmin = 180*60/np.pi


def format_params(params, fit, Nb, arcmin=True):
    """
    Format parameters' array into an easily readable string.

    Parameters
    ----------
    params : np.array
        Array of parameters ordered like [beta, alpha_i, gamma_i] or [psi].
        Angles given in radians. Gamma_i are adimensional.
    fit : str
        Type of fit. Valid string codes: ['pg', 'psi', 'abg', 'ab'].
    Nb : int
        Number of cross array-bands.
    arcmin : bool, optional
        If True, print angles in arcminutes. If False, print angles in degrees.
        The default is True.

    Returns
    -------
    str
        String listing parameter values with units.

    """
    u = 'arcmin' if arcmin else 'deg'
    s = '['
    if fit == 'pg':
        pg = np.rad2deg(params)*60 if arcmin else np.rad2deg(params)
        s += f' {pg:.3f} {u}'
    elif fit == 'abg':
        alpha = np.rad2deg(params[:-Nb]) * 60 if arcmin else np.rad2deg(params[:-Nb])
        gamma = params[Nb+1:]
        for ang in alpha:
            s += f' {ang:.3f} {u},'
        for g in gamma:
            s += f' {g*100:.3f}%,'
    else:
        alpha = np.rad2deg(params)*60 if arcmin else np.rad2deg(params)
        for ang in alpha:
            s += f' {ang:.3f} {u},'
    return s[:-1] + ' ]'



def ang_tag(fit, bands):
    """
    Generate a list indicating which parameters were fitted and in what order.

    Parameters
    ----------
    fit : str
        Type of fit. Valid string codes: ['pg', 'psi', 'abg', 'ab'].
    bands : list
        List containing the identifiers of each array-band.

    Returns
    -------
    var : list of str
        List of parameter names.
    """
    if fit == 'ab':
        var = ['beta']
        for b in bands:
            var += [f'alpha {b}']
            
    elif fit == 'abg':
        var = ['beta']
        for b in bands:
            var += [f'alpha {b}']
        for b in bands:
            var += [f'gamma {b}']
            
    elif fit == 'pi':
        var = []
        for b in bands:
            var += [f'psi {b}']
            
    elif fit == 'pg':
        var = 'global psi'
    return var


def summary(res, show_chi2=True, alpha_pa5_discrepancy=True):
    """
    Print results of the fit on the screen.

    Parameters
    ----------
    res : dict
        Dictionary with fit results.
    show_chi2 : bool, optional
        If True, print the associated chi-square value. The default is True.
    alpha_pa5_discrepancy : bool, optional
        If True, print the different between PA5 rotations. The default is True.

    Raises
    ------
    ValueError
        When the results don't consider different rotations for PA5 but showing
        their discrepancy was still requested.

    Returns
    -------
    None.
    """
    for ii, var in enumerate(res['bands']):
        if 'beta' in var or 'alpha' in var or 'psi' in var:
            s = np.rad2deg(res['sigma/rad'][-1][ii])
            x = np.rad2deg(res['angs/rad'][-1][ii])
            print(f"{var} = ({x:.3f} \pm {s:.3f})deg")
        else:
            s = res['sigma/rad'][-1][ii]*100
            x = res['angs/rad'][-1][ii]*100
            print(f"{var} = ({x:.3f} \pm {s:.3f})%")

    if show_chi2:
        pte = 100*(1 - chi2.cdf(res['chi square'], res['dof']))
        print(f"chi square = {res['chi square']:.0f} for {res['dof']} dof (PTE = {pte:.1f}%)")
        
    if alpha_pa5_discrepancy: 
        idx_090   = 0
        found_090 = False
        while not found_090 and idx_090<len(res['bands']):
            if res['bands'][idx_090] in ['alpha pa5_f090', 'psi pa5_f090']:
                found_090 = True
            else:
                idx_090 += 1
        if not found_090:
            raise ValueError('alpha PA5 f090 not found')  
        idx_150   = 0
        found_150 = False
        while not found_150 and idx_150<len(res['bands']):
            if res['bands'][idx_150] in ['alpha pa5_f150', 'psi pa5_f150']:
                found_150 = True
            else:
                idx_150 += 1
        if not found_150:
            raise ValueError('alpha PA5 f150 not found')
        
        a_090    = np.rad2deg(res['angs/rad'][-1])[idx_090]
        a_150    = np.rad2deg(res['angs/rad'][-1])[idx_150]
        cov      = res['covariance/rad^2'][-1]
        diff_a   = a_150 - a_090
        diff_std = np.rad2deg(np.sqrt(cov[idx_090,idx_090] + cov[idx_150,idx_150] - 2*cov[idx_090,idx_150]))
        print(f'PA5 f150-f090 = ({diff_a:.3f} \pm {diff_std:.3f})deg, {np.abs(diff_a/diff_std):.1f}sigma discrepancy')



class MLE:
    options_allowed = {'channels':   ['EB', 'TB', 'EB+TB', 'EBxTB'],
                       'fit':        ['ab', 'pi', 'pg', 'abg']}
    
    def __init__(self, bands, fit, channel, bandpower,
                 alpha_prior_file=None,
                 gamma_prior_file=None,
                 exclude_auto_spectra=False,
                 include_off_diagonal_obs_blocks=True,
                 ell_diagonal=False):
        """
        Object applying the Minami-Komatsu estimator to ACT DR6 cross-array
        band power spectra. 

        Parameters
        ----------
        bands : list of str
            List containing the identifiers of array-bands to use in the fit.
            Valid codes: 'pa4_f220', 'pa5_f090', 'pa5_f150', 'pa6_f090', 'pa6_f150'
        fit : str
            Parameters to fit. Allowed options are: 
                'ab' for alpha_i + beta 
                'pi' for psi_i (rotation per array-band)
                'pg' for psi_global (global rotation)
                'abg' for alpha_i + beta + gamma_i 
        channel : str
            Information to use in the fit. Allowed options are:
                'EB' for an EB-only fit
                'TB' for a TB-only fit
                'EB+TB' for a fit using EB and TB but treating them as independent
                'EBxTB' for a fit using EB and TB, including their cross-correlations
        bandpower : BandPowers object
            BandPowers object detailing the binning specifications.
        alpha_prior_file : str, optional
            Name tag of the Gaussian prior to apply on miscalibration angles.
            E.g., 'diag0.09-0.09-0.11_pa456'.
            If None, no prior is used. The default is None.
        gamma_prior_file : str, optional
            Name tag of the Gaussian prior to apply on I2P leakage parameters.
            If None, no prior is used. The default is None.
        exclude_auto_spectra : bool, optional
            Array-band auto-spectra (e.g., pa5_f150 x pa5_f150) can be excluded 
            to avoid noise bias. This precaution is not really necessary for 
            ACT DR6. The default is False, hence using auto-spectra.
        include_off_diagonal_obs_blocks : bool, optional
            Whether to include off-diagonal cross-correlations between power 
            spectra of different channels in the covariance matrix (i.e., 
            cov(XY,WZ) when XY=/=WZ). E.g., if False, cov(EB, EB) is used but
            not cov(EB, EE). The default is True, hence including off-diagonal 
            blocks.
        ell_diagonal : bool, optional
            Whether to include off-diagonal bxb' correlations between multipole
            bins. The default is False, hence considering bxb' correlations.

        Raises
        ------
        ValueError
            If the string for channel/fit are not one of the allowed options.

        Returns
        -------
        None.
        """
        #######################################################################
        # Process input options
        #######################################################################
        if channel not in self.options_allowed['channels']:
            raise ValueError(f"'{channel}' is not a valid option")
        if fit not in self.options_allowed['fit']:
            raise ValueError(f"'{fit}' is not a valid option")
        self.channel          = channel
        self.exclude_auto     = exclude_auto_spectra
        self.include_off_diag = include_off_diagonal_obs_blocks
        self.ell_diag         = ell_diagonal
        self.fit              = fit

        #######################################################################
        # Set band configuration
        #######################################################################
        self.bandID = bands
        self.Nbands = len(bands)
        # params to fit
        if fit == 'ab':
            self.Nvar = self.Nbands + 1
        elif fit == 'abg':
            self.Nvar = 2*self.Nbands + 1
        elif fit == 'pi':
            self.Nvar = self.Nbands
        elif fit == 'pg':
            self.Nvar = 1

        # Tricks for fast and efficient indexing
        IJidx = []
        for ii in range(0, self.Nbands, 1):
            for jj in range(0, self.Nbands, 1):
                if jj == ii and self.exclude_auto:
                    pass
                else:
                    IJidx.append((ii, jj))
        self.IJidx = np.array(IJidx, dtype=np.uint8) # Data type valid for <=70 bands

        self.fullSize = self.Nbands*(self.Nbands-1) if self.exclude_auto else self.Nbands**2
        MNidx = []
        for mm in range(0, self.fullSize, 1):
            for nn in range(0, self.fullSize, 1):
                MNidx.append((mm, nn))
        self.MNidx = np.array(MNidx, dtype=np.uint8) # Data type valid for <=70 bands

        # First store things for the maximum Nbins
        self.fullSize *= bandpower.g_Nb
        
        #######################################################################
        # Set binning configuration
        #######################################################################
        self.bandpower = bandpower
        if bandpower.opt != 'fixed':
            self.trimSize = 0
            self.trimIdx = {}
            for (ii, jj) in self.IJidx:
                fi, fj = self.bandID[ii], self.bandID[jj]
                # Keep only the bins common to both frequency bands
                bi, bj = self.bandpower.conf[fi]['b'], self.bandpower.conf[fj]['b']
                nb     = len(set(bi).intersection(set(bj)))
                self.trimIdx[f'({ii}, {jj})'] = {'start': self.trimSize, 'end': self.trimSize+nb}
                self.trimSize += nb
                
        #######################################################################
        # Load priors
        #######################################################################
        if alpha_prior_file != None:
            self.alpha_prior     = True
            prior_dict           = pickle.load(open(f'{prior_dir}/{alpha_prior_file}.pkl', "rb"))
            self.alpha_prior_vec = prior_dict['vec']
            self.alpha_prior_cov = prior_dict['cov']
            # Careful with the pseudo-inverse
            self.invCov_p_a      = np.linalg.pinv(self.alpha_prior_cov)
        else:
            self.alpha_prior     = False
            self.alpha_prior_vec = None
            self.alpha_prior_cov = None
        self.alpha_prior_file    = alpha_prior_file

        if gamma_prior_file != None:
            self.gamma_prior     = True
            prior_dict           = pickle.load(open(f'{prior_dir}/{gamma_prior_file}.pkl', "rb"))
            self.gamma_prior_vec = prior_dict['vec']
            self.gamma_prior_cov = prior_dict['cov']
            # Careful with the pseudo-inverse
            self.invCov_p_g      = np.linalg.pinv(self.gamma_prior_cov)
        else:
            self.gamma_prior     = False
            self.gamma_prior_vec = None
            self.gamma_prior_cov = None
        self.gamma_prior_file = gamma_prior_file
        
        #######################################################################
        # Link covariance matrix, linear system, and chi square calculators
        #######################################################################
        self.cov  = MK_cov(self)
        self.ls   = MK_linear_system(self)
        self.chi2 = MK_chi_square(self)
        self.tag  = self.make_tag()



    def make_tag(self):
        """
        Unequivocal name to identify results saved to disk.

        Returns
        -------
        str
            Tag to identify results.
        """
        band_tag = 'bands'
        band_tag += '0' if 'pa4_f220' in self.bandID else ''
        band_tag += '1' if 'pa5_f090' in self.bandID else ''
        band_tag += '2' if 'pa5_f150' in self.bandID else ''
        band_tag += '3' if 'pa6_f090' in self.bandID else ''
        band_tag += '4' if 'pa6_f150' in self.bandID else ''

        bin_tag = 'bin' + self.bandpower.nlb
        opt     = self.bandpower.opt
        if opt == 'fixed':
            bin_tag += f'{opt}{self.bandpower.g_lmin}-{self.bandpower.g_lmax}'
        else:
            bin_tag += opt

        fit_tag = f'fit{self.fit}'
        
        cov_tag = 'cov'
        cov_tag += '1' if self.exclude_auto else '0'
        cov_tag += '1' if self.ell_diag else '0'
        cov_tag += '1' if self.include_off_diag else '0'

        prior_tag = 'prior'
        prior_tag += self.alpha_prior_file if self.alpha_prior else 'None'
        prior_tag += self.gamma_prior_file if self.gamma_prior else 'None'
        return f'{self.channel}_{band_tag}_{bin_tag}_{fit_tag}_{cov_tag}_{prior_tag}'


    def get_idx(self, mn_pair):
        """
        Index of array-bands stored in each matrix element.
        MATRIX[mm, nn] = cov(X_ii*Y_jj, W_pp*Z_qq) 

        Parameters
        ----------
        mn_pair : tuple of int
            Tuple of mm and nn position in matrix.

        Returns
        -------
        ii : int
            X_ii array-band.
        jj : int
            Y_jj array-band.
        pp : int
            W_pp array-band.
        qq : int
            Z_qq array-band.
        mm : int
            Position of X_ii*Y_jj bands in matrix row.
        nn : int
            Position of W_pp*Z_qq bands in matrix column.
        """
        mm, nn = mn_pair
        ii, jj = self.IJidx[mm]
        pp, qq = self.IJidx[nn]
        return ii, jj, pp, qq, mm, nn


    def load_data(self):
        """
        Load ACT DR6 products to build data vectors and covariance matrix.
        Initially, all TB and EB data are loaded, although only part of them will 
        be used depending on the estimator configuration. This is not optimal,
        but it is not critical in terms of memory usage.
        In terms of running time, loading everying (obs, cmb, vec, cov) in the
        the same loop would be the optimal approach, but it's not critical
        since it doesn't take a long time anyway.

        Returns
        -------
        None.
        """
        self.load_obs_vec()
        self.cov.load_obs_cov()
        if self.fit in ['ab', 'abg']:
            self.load_cmb_vec()


    def load_cmb_vec(self):
        """
        Load LCDM best-fit to ACT DR6 data.

        Returns
        -------
        None.
        """
        # Variables stored like 
        # l Dl_TT Dl_TE Dl_TB Dl_ET Dl_BT Dl_EE Dl_EB Dl_BE Dl_BB
        lcdm_file = np.loadtxt(f'{lcdm_dir}/dr6_lcdm_best_fits/cmb.dat', skiprows=1)
        # Repeat for all bands since beam and pixel window function are deconvolved from data
        # EE
        ee_b_cmb       = bf.bin_cls(np.concatenate(([0.0, 0.0], lcdm_file[:, 6])), self.bandpower.g_info)
        self.Db_EEc_ij = np.ones((self.Nbands, self.Nbands, self.bandpower.g_Nb))*ee_b_cmb
        # BB
        bb_b_cmb       = bf.bin_cls(np.concatenate(([0.0, 0.0], lcdm_file[:, 9])), self.bandpower.g_info)
        self.Db_BBc_ij = np.ones((self.Nbands, self.Nbands, self.bandpower.g_Nb))*bb_b_cmb
        # TE
        te_b_cmb       = bf.bin_cls(np.concatenate(([0.0, 0.0], lcdm_file[:, 2])), self.bandpower.g_info)
        self.Db_TEc_ij = np.ones((self.Nbands, self.Nbands, self.bandpower.g_Nb))*te_b_cmb
        # TT
        tt_b_cmb       = bf.bin_cls(np.concatenate(([0.0, 0.0], lcdm_file[:, 1])), self.bandpower.g_info)
        self.Db_TTc_ij = np.ones((self.Nbands, self.Nbands, self.bandpower.g_Nb))*tt_b_cmb


    def load_obs_vec(self):
        """
        Load observed cross array-band powers from ACT DR6.

        Returns
        -------
        None.
        """
        # Load sacc file
        sacc_file = sacc.Sacc.load_fits(f"{sacc_dir}/v1.0/dr6_data.fits")
        full_vec  = sacc_file.mean

        # At first load the maximum multipole range
        ell, _ = sacc_file.get_ell_cl('cl_ee', 'dr6_pa5_f090_s2', 'dr6_pa5_f090_s2')
        sel    = (ell >= self.bandpower.g_b[0] - 0.5) & (ell <= self.bandpower.g_b[-1]+0.5)

        self.Db_EEo_ij = np.zeros((self.Nbands, self.Nbands, self.bandpower.g_Nb), dtype=np.float64)
        self.Db_BBo_ij = np.zeros((self.Nbands, self.Nbands, self.bandpower.g_Nb), dtype=np.float64)
        self.Db_EBo_ij = np.zeros((self.Nbands, self.Nbands, self.bandpower.g_Nb), dtype=np.float64)
        self.Db_TEo_ij = np.zeros((self.Nbands, self.Nbands, self.bandpower.g_Nb), dtype=np.float64)
        self.Db_TBo_ij = np.zeros((self.Nbands, self.Nbands, self.bandpower.g_Nb), dtype=np.float64)
        for (ii, jj) in self.IJidx:
            fi, fj     = self.bandID[ii], self.bandID[jj]
            s_ee, e_ee = bf.indices('E', fi, 'E', fj)
            s_bb, e_bb = bf.indices('B', fi, 'B', fj)
            s_eb, e_eb = bf.indices('E', fi, 'B', fj)
            s_te, e_te = bf.indices('T', fi, 'E', fj)
            s_tb, e_tb = bf.indices('T', fi, 'B', fj)
            self.Db_EEo_ij[ii, jj, :] = full_vec[s_ee:e_ee][sel]
            self.Db_BBo_ij[ii, jj, :] = full_vec[s_bb:e_bb][sel]
            self.Db_EBo_ij[ii, jj, :] = full_vec[s_eb:e_eb][sel]
            self.Db_TEo_ij[ii, jj, :] = full_vec[s_te:e_te][sel]
            self.Db_TBo_ij[ii, jj, :] = full_vec[s_tb:e_tb][sel]



    def summation(self, opt, v_ij, iC, v_pq):
        """
        Efficient implementation of the summations involved in the construction
        of the linear system.

        Parameters
        ----------
        opt : str
            Which v_ijl * invC_ijpql * v_pql terms to sum up. Allowed options:
                'ijpq' sum only over l
                'ij' sum over pql
                'pq' sum over ijl
                '_' sum all ijpql 
        v_ij : np.ndarray
            Vector containing X_i*Y_j cross array-band spectra 
        iC : np.ndarray
            Inverse cross array-band covariance matrix
        v_pq : np.ndarray
            Vector containing W_p*Z_q cross array-band spectra 

        Raises
        ------
        NotImplementedError
            If opt is not valid option.

        Returns
        -------
        float or np.ndarray
            Summation value. Size will depend on requested opt:
                'ijpq' gives [ii, jj, pp, qq] dimensions
                'ij' gives [ii, jj] dimensions
                'pq' gives [pp, qq] dimensions
                '_' gives a float 
        """
        full_b = self.bandpower.g_b
        x_ijpq = np.zeros((self.Nbands, self.Nbands, self.Nbands, self.Nbands), dtype=np.float64)
        for mn_pair in self.MNidx:
            ii, jj, pp, qq, mm, nn = self.get_idx(mn_pair)
            fi, fj, fp, fq = self.bandID[ii], self.bandID[jj], self.bandID[pp], self.bandID[qq]

            # Select the terms from the vectors
            if self.bandpower.opt == 'fixed':
                v1 = v_ij[ii, jj, :]
                v2 = v_pq[pp, qq, :]
            else:
                # Keep only the bins common to both frequency bands
                bi, bj = self.bandpower.conf[fi]['b'], self.bandpower.conf[fj]['b']
                b_ij   = np.sort(list(set(bi).intersection(set(bj))))
                sel_ij = np.array([b in b_ij for b in full_b])
                v1     = v_ij[ii, jj, sel_ij]

                bp, bq = self.bandpower.conf[fp]['b'], self.bandpower.conf[fq]['b']
                b_pq   = np.sort(list(set(bp).intersection(set(bq))))
                sel_pq = np.array([b in b_pq for b in full_b])
                v2     = v_pq[pp, qq, sel_pq]

            # Select the terms from the matrix
            if self.bandpower.opt == 'fixed':
                s_mm, e_mm = mm*self.bandpower.g_Nb, self.bandpower.g_Nb*(mm+1)
                s_nn, e_nn = nn*self.bandpower.g_Nb, self.bandpower.g_Nb*(nn+1)
            else:
                s_mm, e_mm = self.trimIdx[f'({ii}, {jj})']['start'], self.trimIdx[f'({ii}, {jj})']['end']
                s_nn, e_nn = self.trimIdx[f'({pp}, {qq})']['start'], self.trimIdx[f'({pp}, {qq})']['end']

            x_ijpq[ii, jj, pp, qq] = np.matmul(v1, np.matmul(iC[s_mm:e_mm, s_nn:e_nn], v2))

        if opt == 'ijpq':
            return x_ijpq
        elif opt == 'ij':
            return np.sum(np.sum(x_ijpq, axis=3), axis=2)
        elif opt == 'pq':
            return np.sum(np.sum(x_ijpq, axis=1), axis=0)
        elif opt == '_':
            return np.sum(x_ijpq)
        else:
            raise NotImplementedError(f'{opt} is not a valid option')
    

    def solve(self, save=False, verbose=True):
        """
        Iterative semi-analytical solution to find the maximum-likelihood (or
        maximum posterior, when including priors) parameters.

        Parameters
        ----------
        save : bool, optional
            Whether to save the results to disk. The default is False.
        verbose: bool, optional
            Whether to print additional information during execution. The default is True.
            
        Returns
        -------
        results : dict
            Dictionary with results.
        """
        # Angles start at 0 and are saved in radians
        # Amplitudes start at 0 (doesn't matter, they don't enter the covariance)
        if self.fit == 'pg':
            ang_list = [0]
            std_list = [0]
            cov_list = [0]
        else:
            ang_list = [np.zeros(self.Nvar, dtype=np.float64)]
            std_list = [np.zeros(self.Nvar, dtype=np.float64)]
            cov_list = [np.zeros((self.Nvar, self.Nvar), dtype=np.float64)]

        if verbose: print('Solving linear system ...')
        converged = False
        niter     = 0
        while not converged:
            if verbose: print(f'\t iter {niter+1}:')
            # Calculate covariance (internally chooses channel)
            cov_now    = self.cov.build_cov(ang_list[niter])
            invCov_now = np.linalg.inv(cov_now)
            del cov_now

            # Solve system for this iteration (internally chooses channel)
            if self.fit == 'ab':
                ang_now, cov_now, std_now = self.ls.solve_ab(invCov_now)
            elif self.fit == 'abg':
                ang_now, cov_now, std_now = self.ls.solve_abg(invCov_now)
            elif self.fit == 'pi':
                ang_now, cov_now, std_now = self.ls.solve_pi(invCov_now)
            elif self.fit == 'pg':
                ang_now, cov_now, std_now = self.ls.solve_pg(invCov_now)

            if verbose: print(f'{format_params(ang_now, self.fit, self.Nbands)}')
            if np.any(np.isnan(std_now)):
                # Stop iterating
                warnings.warn(f'\t NaN in covariance. Aborting at iteration {niter+1}.')
                converged = True
            else:
                # Evaluate convergence of the iterative calculation
                # Regulate tolerance depending on the sensitivity to angle measurement
                tol = 0.5 if np.min(std_now)*rad2arcmin > 0.5 else 0.1
                # Use alpha + beta sum as convergence criterion
                # Difference with i-1
                if self.fit == 'ab':
                    apb_now = ang_now[:1] + ang_now[1:]
                    apb_1   = ang_list[niter][:1] + ang_list[niter][1:]
                elif self.fit == 'abg':
                    apb_now = ang_now[:1] + ang_now[1:-self.Nbands]
                    apb_1   = ang_list[niter][:1] + ang_list[niter][1:-self.Nbands]
                else:
                    apb_now = ang_now
                    apb_1   = ang_list[niter]
                c1 = np.abs(apb_now - apb_1)*rad2arcmin >= tol
                if np.sum(c1) < 1 or niter > max_iter:
                    converged = True
                elif niter > 0:
                    # Difference with i-2
                    if self.fit == 'ab':
                        apb_2 = ang_list[niter-1][:1] + ang_list[niter-1][1:]
                    elif self.fit == 'abg':
                        apb_2 = ang_list[niter-1][:1] + ang_list[niter-1][1:-self.Nbands]
                    else:
                        apb_2 = ang_list[niter-1]
                    c2 = np.abs(apb_now - apb_2)*rad2arcmin >= tol
                    if np.sum(c2) < 1:
                        converged = True

            # Store results
            ang_list.append(ang_now)
            cov_list.append(cov_now)
            std_list.append(std_now)

            niter = niter+1

        if verbose: print('Done!')

        # Calculate chi-square for the final result (internally chooses channel)
        if self.fit == 'ab':
            final_chi2 = self.chi2.calc_ab(ang_list[-1], self.cov.build_cov(ang_list[-1]))
        elif self.fit == 'abg':
            final_chi2 = self.chi2.calc_abg(ang_list[-1], self.cov.build_cov(ang_list[-1]))
        elif self.fit == 'pi':
            final_chi2 = self.chi2.calc_pi(ang_list[-1], self.cov.build_cov(ang_list[-1]))
        elif self.fit == 'pg':
            final_chi2 = self.chi2.calc_pg(ang_list[-1], self.cov.build_cov(ang_list[-1]))

        results = {'angs/rad': np.array(ang_list),
                   'sigma/rad': np.array(std_list),
                   'covariance/rad^2': np.array(cov_list),
                   'bands': ang_tag(self.fit, self.bandID),
                   'chi square': final_chi2, 'dof': self.chi2.dof()}

        if save:
            with open(f"{res_dir}/{self.tag}.pkl", 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return results


    def read_result(self):
        """
        Load results from disk.

        Raises
        ------
        FileNotFoundError
            If this particular estimator was not run beforehand.

        Returns
        -------
        dict
            Dictionary with the result of the fit.
        """
        file = f"{res_dir}/{self.tag}.pkl"
        if os.path.isfile(file):
            return pickle.load(open(file, "rb"))
        else:
            raise FileNotFoundError(f'{file} does not exist')

