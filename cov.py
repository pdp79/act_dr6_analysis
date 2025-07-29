#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################################################
# Covariance matrix associated to the Minami-Komatsu estimator.
#
# Excuse the weird naming of variables. That's the shorthand I used when 
# deriving equations on pen and paper.
#
# @author: Patricia Diego-Palazuelos (diegop@mpa-garching.mpg.de)
#####################################################################

import numpy as np
import sacc
from act_dr6_analysis import sacc_dir
from act_dr6_analysis.binning import indices

class MK_cov:
    
    def __init__(self, mle):
        """
        Object calculating the cross array-band covariance matrix of the 
        Minami-Komatsu estimator.

        Parameters
        ----------
        mle : MLE object
            Maximum-likelihood estimator the covariance matrix is associated with.

        Returns
        -------
        None.
        """
        self.mle = mle 
        #######################################################################
        ### Duplicate some of the information for easy reference
        ### It's not memory heavy but I know it's a bad practice
        #######################################################################
        self.bp         = mle.bandpower
        self.fullSize   = mle.fullSize
        self.trimSize   = mle.trimSize
        self.ch         = mle.channel
        self.off_diag   = mle.include_off_diag
        self.ell_diag   = mle.ell_diag
        self.fit        = mle.fit
        self.bands      = mle.bandID
        self.Nb         = mle.Nbands
        #######################################################################
        ### Tricks for fast indexing
        #######################################################################
        # data type valid for <=70 bands, optimizing memory use
        self.MNBi = np.zeros((self.fullSize, self.fullSize), dtype=np.uint8)
        self.MNBj = np.zeros((self.fullSize, self.fullSize), dtype=np.uint8)
        self.MNBp = np.zeros((self.fullSize, self.fullSize), dtype=np.uint8)
        self.MNBq = np.zeros((self.fullSize, self.fullSize), dtype=np.uint8)
        for mn_pair in mle.MNidx:
            ii, jj, pp, qq, mm, nn = mle.get_idx(mn_pair)
            s_mm, e_mm = mm*self.bp.g_Nb, self.bp.g_Nb*(mm+1)
            s_nn, e_nn = nn*self.bp.g_Nb, self.bp.g_Nb*(nn+1)
            self.MNBi[s_mm:e_mm, s_nn:e_nn] = ii
            self.MNBj[s_mm:e_mm, s_nn:e_nn] = jj
            self.MNBp[s_mm:e_mm, s_nn:e_nn] = pp
            self.MNBq[s_mm:e_mm, s_nn:e_nn] = qq
        
        
    #######################################################################
    ### Load elements of the covariance
    #######################################################################
    
    def load_obs_cov(self):
        """
        Load input ACT DR6 covariance blocks, ordering them in the proper
        structure for this code. Initially, all TB and EB data are loaded, 
        although only part of them will be used depending on the estimator 
        configuration. This is not optimal,  but it is not critical in 
        terms of memory usage

        Returns
        -------
        None.
        """
        dt        = np.float64
        # load sacc file
        sacc_file = sacc.Sacc.load_fits(f"{sacc_dir}/v1.0/dr6_data.fits")
        full_cov  = sacc_file.covariance.covmat
        # first load maximum multipole range
        ell, _    = sacc_file.get_ell_cl('cl_ee', 'dr6_pa5_f090_s2', 'dr6_pa5_f090_s2')
        sel       = (ell >= self.bp.g_b[0] - 0.5) & (ell <= self.bp.g_b[-1]+0.5)

        self.cov_EEo_EEo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_EEo_EBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_EEo_BBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_EBo_EEo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_EBo_EBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_EBo_BBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_BBo_EEo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_BBo_EBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_BBo_BBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        ##################
        self.cov_TEo_TEo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_TEo_TBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_TBo_TEo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_TBo_TBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        ##################
        self.cov_TBo_EEo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_TBo_EBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_TBo_BBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_TEo_EEo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_TEo_EBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_TEo_BBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        ##################
        self.cov_EEo_TBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_EBo_TBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_BBo_TBo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_EEo_TEo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_EBo_TEo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        self.cov_BBo_TEo = np.zeros((self.fullSize, self.fullSize), dtype=dt)
        for mn_pair in self.mle.MNidx:
            ii, jj, pp, qq, mm, nn = self.mle.get_idx(mn_pair)
            fi, fj, fp, fq = self.bands[ii], self.bands[jj], self.bands[pp], self.bands[qq]
            mat = np.eye(self.bp.g_Nb) if self.ell_diag else np.ones((self.bp.g_Nb, self.bp.g_Nb))

            s_mm, e_mm       = mm*self.bp.g_Nb, self.bp.g_Nb*(mm+1)
            s_ee_mm, e_ee_mm = indices('E', fi, 'E', fj)
            s_bb_mm, e_bb_mm = indices('B', fi, 'B', fj)
            s_eb_mm, e_eb_mm = indices('E', fi, 'B', fj)
            s_te_mm, e_te_mm = indices('T', fi, 'E', fj)
            s_tb_mm, e_tb_mm = indices('T', fi, 'B', fj)
            idx_ee_mm = np.arange(s_ee_mm, e_ee_mm)
            idx_bb_mm = np.arange(s_bb_mm, e_bb_mm)
            idx_eb_mm = np.arange(s_eb_mm, e_eb_mm)
            idx_te_mm = np.arange(s_te_mm, e_te_mm)
            idx_tb_mm = np.arange(s_tb_mm, e_tb_mm)

            s_nn, e_nn       = nn*self.bp.g_Nb, self.bp.g_Nb*(nn+1)
            s_ee_nn, e_ee_nn = indices('E', fp, 'E', fq)
            s_bb_nn, e_bb_nn = indices('B', fp, 'B', fq)
            s_eb_nn, e_eb_nn = indices('E', fp, 'B', fq)
            s_te_nn, e_te_nn = indices('T', fp, 'E', fq)
            s_tb_nn, e_tb_nn = indices('T', fp, 'B', fq)
            idx_ee_nn = np.arange(s_ee_nn, e_ee_nn)
            idx_bb_nn = np.arange(s_bb_nn, e_bb_nn)
            idx_eb_nn = np.arange(s_eb_nn, e_eb_nn)
            idx_te_nn = np.arange(s_te_nn, e_te_nn)
            idx_tb_nn = np.arange(s_tb_nn, e_tb_nn)

            self.cov_EEo_EEo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_ee_mm[sel], idx_ee_nn[sel])]*mat
            self.cov_EBo_EBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_eb_mm[sel], idx_eb_nn[sel])]*mat
            self.cov_BBo_BBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_bb_mm[sel], idx_bb_nn[sel])]*mat
            self.cov_TEo_TEo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_te_mm[sel], idx_te_nn[sel])]*mat
            self.cov_TBo_TBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_tb_mm[sel], idx_tb_nn[sel])]*mat
            if self.off_diag:
                self.cov_EEo_EBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_ee_mm[sel], idx_eb_nn[sel])]*mat
                self.cov_EEo_BBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_ee_mm[sel], idx_bb_nn[sel])]*mat
                self.cov_EBo_EEo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_eb_mm[sel], idx_ee_nn[sel])]*mat
                self.cov_EBo_BBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_eb_mm[sel], idx_bb_nn[sel])]*mat
                self.cov_BBo_EEo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_bb_mm[sel], idx_ee_nn[sel])]*mat
                self.cov_BBo_EBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_bb_mm[sel], idx_eb_nn[sel])]*mat
                self.cov_TEo_TBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_te_mm[sel], idx_tb_nn[sel])]*mat
                self.cov_TBo_TEo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_tb_mm[sel], idx_te_nn[sel])]*mat
            if self.ch == 'EBxTB':
                self.cov_TBo_EEo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_tb_mm[sel], idx_ee_nn[sel])]*mat
                self.cov_TBo_EBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_tb_mm[sel], idx_eb_nn[sel])]*mat
                self.cov_TBo_BBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_tb_mm[sel], idx_bb_nn[sel])]*mat
                self.cov_TEo_EEo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_te_mm[sel], idx_ee_nn[sel])]*mat
                self.cov_TEo_EBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_te_mm[sel], idx_eb_nn[sel])]*mat
                self.cov_TEo_BBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_te_mm[sel], idx_bb_nn[sel])]*mat
                ##################
                self.cov_EEo_TBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_ee_mm[sel], idx_tb_nn[sel])]*mat
                self.cov_EBo_TBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_eb_mm[sel], idx_tb_nn[sel])]*mat
                self.cov_BBo_TBo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_bb_mm[sel], idx_tb_nn[sel])]*mat
                self.cov_EEo_TEo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_ee_mm[sel], idx_te_nn[sel])]*mat
                self.cov_EBo_TEo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_eb_mm[sel], idx_te_nn[sel])]*mat
                self.cov_BBo_TEo[s_mm:e_mm, s_nn:e_nn] = full_cov[np.ix_(idx_bb_mm[sel], idx_te_nn[sel])]*mat


    #######################################################################
    # Build the covariance for every iteration
    #######################################################################

    def build_cov(self, params):
        """
        Calculate cross array-band covariance matrix for the input parameters.
        The function internally combines the information from an EB-only fit, 
        TB-only fit, fit using both EB and TB but treating them as independent 
        ('EB+TB'), and the fit including their cross-correlations ('EBxTB').

        Parameters
        ----------
        params : np.array
            Array containing the [beta, alpha_i, gamma_i] or [psi] or psi_g parameters
            Angles are in radians. Gamma_i are adimensional.

        Returns
        -------
        cov : np.ndarray
            Cross array-band covariance matrix
        """
        # If gamma is not going to change the covariance, the easiest is to
        # remove it here
        if self.fit == 'abg':
            # variables ordered like beta, alpha_i, gamma_i
            params = params[:-self.Nb]

        if self.ch == 'EB':
            cov = self.__build_cov_EBxEB__(params)
        elif self.ch == 'TB':
            cov = self.__build_cov_TBxTB__(params)
        elif self.ch == 'EB+TB':
            side = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            cov  = np.zeros((2*side, 2*side), dtype=np.float64)
            cov[:side, :side] = self.__build_cov_EBxEB__(params)
            cov[side:, side:] = self.__build_cov_TBxTB__(params)
        elif self.ch == 'EBxTB':
            side = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            cov  = np.zeros((2*side, 2*side), dtype=np.float64)
            cov[:side, :side] = self.__build_cov_EBxEB__(params)
            cov[:side, side:] = self.__build_cov_EBxTB__(params)
            cov[side:, :side] = self.__build_cov_TBxEB__(params)
            cov[side:, side:] = self.__build_cov_TBxTB__(params)
        return cov


    def __build_cov_EBxTB__(self, params):
        """
        Internal function calculating the EBxTB block of the cross array-band
        covariance matrix

        Parameters
        ----------
        params : np.array
            Array containing the [beta, alpha_i, gamma_i] or [psi] or psi_g parameters
            Angles are in radians. Gamma_i are adimensional.

        Returns
        -------
        np.ndarray
            EBxTB block of the cross array-band covariance matrix
        """
        if self.fit == 'ab' or self.fit == 'abg':
            alpha = params[1:]
        elif self.fit == 'pi':
            alpha = params
        elif self.fit == 'pg':
            alpha = np.repeat(params, self.Nb)
        ai  = alpha[self.MNBi]
        aj  = alpha[self.MNBj]
        aq  = alpha[self.MNBq]
        Aij = np.sin(4*aj)/(np.cos(4*ai)+np.cos(4*aj))
        Bij = np.sin(4*ai)/(np.cos(4*ai)+np.cos(4*aj))
        Dq  = np.tan(2*aq)
        # observed*observed
        cov  = self.cov_EBo_TBo.copy() - Bij*Dq*self.cov_BBo_TEo.copy()
        cov -= Aij*self.cov_EEo_TBo.copy() + Dq*self.cov_EBo_TEo.copy()
        cov += Bij*self.cov_BBo_TBo.copy() + Aij*Dq*self.cov_EEo_TEo.copy()
        # Exclude the extra bins
        if self.bp.opt == 'fixed':
            return cov
        else:
            return self.trim_cov(cov)


    def __build_cov_TBxEB__(self, params):
        """
        Internal function calculating the TBxEB block of the cross array-band
        covariance matrix

        Parameters
        ----------
        params : np.array
            Array containing the [beta, alpha_i, gamma_i] or [psi] or psi_g parameters
            Angles are in radians. Gamma_i are adimensional.

        Returns
        -------
        np.ndarray
            TBxEB block of the cross array-band covariance matrix
        """
        if self.fit == 'ab' or self.fit == 'abg':
            alpha = params[1:]
        elif self.fit == 'pi':
            alpha = params
        elif self.fit == 'pg':
            alpha = np.repeat(params, self.Nb)
        aj  = alpha[self.MNBj]
        ap  = alpha[self.MNBp]
        aq  = alpha[self.MNBq]
        Apq = np.sin(4*aq)/(np.cos(4*ap)+np.cos(4*aq))
        Bpq = np.sin(4*ap)/(np.cos(4*ap)+np.cos(4*aq))
        Dj  = np.tan(2*aj)
        # observed*observed
        cov  = self.cov_TBo_EBo.copy() - Bpq*Dj*self.cov_TEo_BBo.copy()
        cov -= Apq*self.cov_TBo_EEo.copy() + Dj*self.cov_TEo_EBo.copy()
        cov += Bpq*self.cov_TBo_BBo.copy() + Apq*Dj*self.cov_TEo_EEo.copy()
        # Exclude the extra bins from the matrix I return
        if self.bp.opt == 'fixed':
            return cov
        else:
            return self.trim_cov(cov)


    def __build_cov_TBxTB__(self, params):
        """
        Internal function calculating the TBxTB block of the cross array-band
        covariance matrix

        Parameters
        ----------
        params : np.array
            Array containing the [beta, alpha_i, gamma_i] or [psi] or psi_g parameters
            Angles are in radians. Gamma_i are adimensional..

        Returns
        -------
        np.ndarray
            TBxTB block of the cross array-band covariance matrix
        """
        if self.fit == 'ab' or self.fit == 'abg':
            alpha = params[1:]
        elif self.fit == 'pi':
            alpha = params
        elif self.fit == 'pg':
            alpha = np.repeat(params, self.Nb)
        aj = alpha[self.MNBj]
        aq = alpha[self.MNBq]
        Tj = np.tan(2*aj)
        Tq = np.tan(2*aq)
        # observed*observed
        cov  = self.cov_TBo_TBo.copy() + Tq*Tj*self.cov_TEo_TEo.copy()
        cov -= Tq*self.cov_TBo_TEo.copy() + Tj*self.cov_TEo_TBo.copy()
        # Exclude the extra bins
        if self.bp.opt == 'fixed':
            return cov
        else:
            return self.trim_cov(cov)


    def __build_cov_EBxEB__(self, params):
        """
        Internal function calculating the EBxEB block of the cross array-band
        covariance matrix

        Parameters
        ----------
        params : np.array
            Array containing the [beta, alpha_i, gamma_i] or [psi] or psi_g parameters
            Angles are in radians. Gamma_i are adimensional.

        Returns
        -------
        np.ndarray
            EBxEB block of the cross array-band covariance matrix
        """
        if self.fit == 'ab' or self.fit == 'abg':
            alpha = params[1:]
        elif self.fit == 'pi':
            alpha = params
        elif self.fit == 'pg':
            alpha = np.repeat(params, self.Nb)
        ai  = alpha[self.MNBi]
        aj  = alpha[self.MNBj]
        ap  = alpha[self.MNBp]
        aq  = alpha[self.MNBq]
        Aij = np.sin(4*aj)/(np.cos(4*ai)+np.cos(4*aj))
        Bij = np.sin(4*ai)/(np.cos(4*ai)+np.cos(4*aj))
        Apq = np.sin(4*aq)/(np.cos(4*ap)+np.cos(4*aq))
        Bpq = np.sin(4*ap)/(np.cos(4*ap)+np.cos(4*aq))
        # observed*observed
        cov  = self.cov_EBo_EBo.copy()
        cov -= Apq*self.cov_EBo_EEo.copy() + Aij*self.cov_EEo_EBo.copy()
        cov += Bpq*self.cov_EBo_BBo.copy() + Bij*self.cov_BBo_EBo.copy()
        cov += Apq*Aij*self.cov_EEo_EEo.copy() + Bpq*Bij*self.cov_BBo_BBo.copy()
        cov -= Aij*Bpq*self.cov_EEo_BBo.copy() + Bij*Apq*self.cov_BBo_EEo.copy()
        # Exclude the extra bins
        if self.bp.opt == 'fixed':
            return cov
        else:
            return self.trim_cov(cov)


    def trim_cov(self, cov):
        """
        Remove the undesired multipole bins from the covariance matrix.

        Parameters
        ----------
        cov : np.ndarray
            Input covariance matrix

        Returns
        -------
        np.ndarray
            Trimmed covariance matrix 
        """
        trimCov = np.zeros((self.trimSize, self.trimSize), dtype=np.float64)
        full_b  = self.bp.g_b
        # Go block by block getting only the bins that you want
        for mn_pair in self.mle.MNidx:
            ii, jj, pp, qq, mm, nn = self.mle.get_idx(mn_pair)

            # Select block from the original full matrix
            block = cov[mm*self.bp.g_Nb:self.bp.g_Nb*(mm+1), nn*self.bp.g_Nb:self.bp.g_Nb*(nn+1)]

            # Keep only the bins common to both frequency bands
            bi, bj = self.bp.conf[self.bands[ii]]['b'], self.bp.conf[self.bands[jj]]['b']
            b_ij   = np.sort(list(set(bi).intersection(set(bj))))
            sel_ij = np.array([b in b_ij for b in full_b])

            bp, bq = self.bp.conf[self.bands[pp]]['b'], self.bp.conf[self.bands[qq]]['b']
            b_pq   = np.sort(list(set(bp).intersection(set(bq))))
            sel_pq = np.array([b in b_pq for b in full_b])

            # Save them in the new position in the trimmed matrix
            # Which I have previously calculated
            st_mm, et_mm = self.mle.trimIdx[f'({ii}, {jj})']['start'], self.mle.trimIdx[f'({ii}, {jj})']['end']
            st_nn, et_nn = self.mle.trimIdx[f'({pp}, {qq})']['start'], self.mle.trimIdx[f'({pp}, {qq})']['end']
            # print(f'{block.shape} -> {trim_block.shape}')
            trimCov[st_mm:et_mm, st_nn:et_nn] = block[np.ix_(sel_ij, sel_pq)]
        return trimCov
