#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################################################
# Linear system that analytically calculates the maximum-likelihood
# solution for the Minami-Komatsu estimator. The linear system is 
# derived from the optimization over log(likelihood)+log(prior).
# Hence, it actually returns the maximum of the posterior distribution. 
#
# Uncertainties calculated within the Fisher matrix approximation around
# the maximum.
#
# Excuse the weird naming of variables. That's the shorthand I used when 
# deriving equations on pen and paper.
#
# @author: Patricia Diego-Palazuelos (diegop@mpa-garching.mpg.de)
#####################################################################

import numpy as np

class MK_linear_system:
    
    def __init__(self, mle):
        """
        Object solving the linear system associated with the Minami-Komatsu
        estimator.

        Parameters
        ----------
        mle : MLE object
            Maximum-likelihood estimator the linear system is associated with.

        Returns
        -------
        None.
        """
        self.mle = mle
        #######################################################################
        ### Duplicate some of the information for easy reference
        ### It's not memory heavy but I know it's a bad practice
        #######################################################################
        self.bp       = mle.bandpower
        self.fullSize = mle.fullSize
        self.trimSize = mle.trimSize
        self.ch       = mle.channel
        self.bands    = mle.bandID
        self.Nb       = mle.Nbands
        self.Nv       = mle.Nvar
        self.a_prior  = mle.alpha_prior
        if self.a_prior:
            self.a_vec    = mle.alpha_prior_vec
            self.a_invcov = mle.invCov_p_a
        self.g_prior  = mle.gamma_prior
        if self.g_prior:
            self.g_vec    = mle.gamma_prior_vec
            self.g_invcov = mle.invCov_p_g        
        
        
    #######################################################################
    ### Fit to alpha+beta
    #######################################################################       

    def solve_ab(self, iC):
        """
        Solve linear system fitting alpha+beta. The function internally
        combines the information from an EB-only fit, TB-only fit, fit using
        both EB and TB but treating them as independent ('EB+TB'), and the fit
        including their cross-correlations ('EBxTB').

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        ang : np.array
            Best-fit [beta, alpha_i] values in radians
        cov : np.ndarray
            Covariance between variables
        std : np.array
            Sigma of variables
        """
        # Build the basic linear system
        if self.ch == 'EB':
            # iC is already the block you want
            sys_mat, ind_term = self.__sys_elem_ab_ebxeb__(iC)
        elif self.ch == 'TB':
            # iC is already the block you want
            sys_mat, ind_term = self.__sys_elem_ab_tbxtb__(iC)
        elif self.ch == 'EB+TB':
            side = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            sm_ebxeb, it_ebxeb = self.__sys_elem_ab_ebxeb__(iC[:side, :side])
            sm_tbxtb, it_tbxtb = self.__sys_elem_ab_tbxtb__(iC[side:, side:])
            sys_mat  = sm_ebxeb + sm_tbxtb
            ind_term = it_ebxeb + it_tbxtb
        elif self.ch == 'EBxTB':
            side = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            sm_ebxeb, it_ebxeb = self.__sys_elem_ab_ebxeb__(iC[:side, :side])
            sm_tbxtb, it_tbxtb = self.__sys_elem_ab_tbxtb__(iC[side:, side:])
            sm_ebxtb, it_ebxtb = self.__sys_elem_ab_ebxtb__(iC[:side, side:])
            sm_tbxeb, it_tbxeb = self.__sys_elem_ab_tbxeb__(iC[side:, :side])
            sys_mat  = sm_ebxeb + sm_tbxtb + sm_ebxtb + sm_tbxeb
            ind_term = it_ebxeb + it_tbxtb + it_ebxtb + it_tbxeb
        # Add priors if needed
        if self.a_prior:
            for ii in range(self.Nb):
                # alpha_i
                ind_term[ii+1] += np.sum(self.a_vec * self.a_invcov[ii, :])
                for jj in range(self.Nb):
                    # alpha_i - alpha_j
                    sys_mat[ii+1, jj+1] += self.a_invcov[ii, jj]
        # Solve Ax=B and calculate Fisher matrix
        ang = np.linalg.solve(sys_mat, ind_term)
        # Prefactor are chosen so that the system matrix equals the Fisher matrix
        cov = np.linalg.inv(sys_mat)
        std = np.sqrt(np.diagonal(cov))
        return ang, cov, std
    

    def __sys_elem_ab_ebxeb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the EBxEB contribution to the alpha+beta fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        B_ijpq     = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_BBo_ij)
        E_ijpq     = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq     = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EEo_ij)
        D_ij       = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_EBo_ij)
        H_ij       = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_EBo_ij)
        tau_ij     = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_EEc_ij)
        varphi_ij  = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_BBc_ij)
        ene_ij     = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_EEc_ij)
        epsilon_ij = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_BBc_ij)
        C          = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_BBc_ij)
        F          = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_EEc_ij)
        G          = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_BBc_ij)
        O          = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_EEc_ij)
        P          = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_BBc_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as beta, alpha_i
        sys_mat[0, 0] = 4*(G + F - 2*C)  # beta - beta
        ind_term[0]   = 2*(O - P)       # beta
        for ii in range(self.Nb):
            # beta - alpha_i
            b_a = np.sum(tau_ij[:, ii]) + np.sum(epsilon_ij[ii, :]) - np.sum(varphi_ij[:, ii]) - np.sum(ene_ij[ii, :])
            sys_mat[   0, ii+1] += 4*b_a
            sys_mat[ii+1,    0] += 4*b_a
            # alpha_i
            ind_term[ii+1] += 2*(np.sum(D_ij[:, ii]) - np.sum(H_ij[ii, :]))
            for jj in range(self.Nb):
                # alpha_i - alpha_j terms
                aux1 = np.sum(E_ijpq[:, jj, :, ii]) + np.sum(E_ijpq[:, ii, :, jj])
                aux2 = np.sum(B_ijpq[jj, :, ii, :]) + np.sum(B_ijpq[ii, :, jj, :])
                aux3 = np.sum(I_ijpq[jj, :, :, ii]) + np.sum(I_ijpq[ii, :, :, jj])
                sys_mat[ii+1, jj+1] += 2*(aux1 + aux2 - 2*aux3)
        return sys_mat, ind_term
    
    

    def __sys_elem_ab_tbxtb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the TBxTB contribution to the alpha+beta fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TEo_ij)
        D_ij   = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_TBo_ij)
        tau_ij = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_TEc_ij)
        C      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_TEc_ij)
        O      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_TEc_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as beta, alpha_i
        sys_mat[0, 0] = 4*C  # beta - beta
        ind_term[0]   = 2*O  # beta
        for ii in range(self.Nb):
            # beta - alpha_i
            sys_mat[   0, ii+1] += 4*np.sum(tau_ij[:, ii])
            sys_mat[ii+1,    0] += 4*np.sum(tau_ij[:, ii])
            # alpha_i
            ind_term[ii+1] += 2*np.sum(D_ij[:, ii])
            for jj in range(self.Nb):
                # alpha_i - alpha_j terms
                sys_mat[ii+1, jj+1] += 2*np.sum(I_ijpq[:, jj, :, ii])
                sys_mat[ii+1, jj+1] += 2*np.sum(I_ijpq[:, ii, :, jj])
        return sys_mat, ind_term


    def __sys_elem_ab_tbxeb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the TBxEB contribution to the alpha+beta fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_BBo_ij)
        Z_ij   = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_EBo_ij)
        W_ij   = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_EEc_ij)
        R_ij   = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_BBc_ij)
        X_pq   = self.mle.summation('pq',   self.mle.Db_TBo_ij, iC, self.mle.Db_EEo_ij)
        Y_pq   = self.mle.summation('pq',   self.mle.Db_TBo_ij, iC, self.mle.Db_BBo_ij)
        S_pq   = self.mle.summation('pq',   self.mle.Db_TEc_ij, iC, self.mle.Db_EEo_ij)
        T_pq   = self.mle.summation('pq',   self.mle.Db_TEc_ij, iC, self.mle.Db_BBo_ij)
        E      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_EEc_ij)
        B      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_BBc_ij)
        C      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_EBo_ij)
        D      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_EEc_ij)
        F      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_BBc_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as beta, alpha_i
        sys_mat[0, 0] = 4*(D - F)  # beta - beta
        ind_term[0]   = C + E - B  # beta
        for ii in range(self.Nb):
            # beta - alpha_i
            b_a = np.sum(W_ij[:, ii]) + np.sum(S_pq[:, ii]) - np.sum(R_ij[:, ii]) - np.sum(T_pq[ii, :])
            sys_mat[   0, ii+1] += 2*b_a
            sys_mat[ii+1,    0] += 2*b_a
            # alpha_i
            ind_term[ii+1] += np.sum(X_pq[:, ii]) + np.sum(Z_ij[:, ii]) - np.sum(Y_pq[ii, :])
            for jj in range(self.Nb):
                # alpha_i - alpha_j terms
                aux1 = np.sum(H_ijpq[:, jj, :, ii]) + np.sum(H_ijpq[:, ii, :, jj])
                aux2 = np.sum(I_ijpq[:, ii, jj, :]) + np.sum(I_ijpq[:, jj, ii, :])
                sys_mat[ii+1, jj+1] += 2*(aux1 - aux2)
        return sys_mat, ind_term


    def __sys_elem_ab_ebxtb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the EBxTB contribution to the alpha+beta fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TEo_ij)
        Z_pq   = self.mle.summation('pq',   self.mle.Db_EBo_ij, iC, self.mle.Db_TEo_ij)
        W_pq   = self.mle.summation('pq',   self.mle.Db_EEc_ij, iC, self.mle.Db_TEo_ij)
        R_pq   = self.mle.summation('pq',   self.mle.Db_BBc_ij, iC, self.mle.Db_TEo_ij)
        X_ij   = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_TBo_ij)
        Y_ij   = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_TBo_ij)
        S_ij   = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_TEc_ij)
        T_ij   = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_TEc_ij)
        E      = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_TBo_ij)
        B      = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_TBo_ij)
        C      = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_TEc_ij)
        D      = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_TEc_ij)
        F      = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_TEc_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as beta, alpha_i
        sys_mat[0, 0] = 4*(D - F)  # beta - beta
        ind_term[0]   = C + E - B  # beta
        for ii in range(self.Nb):
            # beta - alpha_i
            b_a = np.sum(W_pq[:, ii]) + np.sum(S_ij[:, ii]) - np.sum(R_pq[:, ii]) - np.sum(T_ij[ii, :])
            sys_mat[   0, ii+1] += 2*b_a
            sys_mat[ii+1,    0] += 2*b_a
            # alpha_i
            ind_term[ii+1] += np.sum(X_ij[:, ii]) + np.sum(Z_pq[:, ii]) - np.sum(Y_ij[ii, :])
            for jj in range(self.Nb):
                # alpha_i - alpha_j terms
                aux1 = np.sum(H_ijpq[:, jj, :, ii]) + np.sum(H_ijpq[:, ii, :, jj])
                aux2 = np.sum(I_ijpq[:, ii, jj, :]) + np.sum(I_ijpq[:, jj, ii, :])
                sys_mat[ii+1, jj+1] += 2*(aux1 - aux2)
        return sys_mat, ind_term


    #######################################################################
    ### Fit to alpha+beta+gamma
    ####################################################################### 

    def solve_abg(self, iC):
        """
        Solve linear system fitting alpha+beta+gamma. The function internally
        combines the information from an EB-only fit, TB-only fit, fit using
        both EB and TB but treating them as independent ('EB+TB'), and the fit
        including their cross-correlations ('EBxTB').

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        ang : np.array
            Best-fit [beta, alpha_i, gamma_i] values. 
            Angles are in radians. Gamma_i are adimensional
        cov : np.ndarray
            Covariance between variables
        std : np.array
            Sigma of variables
        """
        # Build the basic linear system
        if self.ch == 'EB':
            # iC is already the block you want
            sys_mat, ind_term = self.__sys_elem_abg_ebxeb__(iC)
        elif self.ch == 'TB':
            # iC is already the block you want
            sys_mat, ind_term = self.__sys_elem_abg_tbxtb__(iC)
        elif self.ch == 'EB+TB':
            side = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            sm_ebxeb, it_ebxeb = self.__sys_elem_abg_ebxeb__(iC[:side, :side])
            sm_tbxtb, it_tbxtb = self.__sys_elem_abg_tbxtb__(iC[side:, side:])
            sys_mat  = sm_ebxeb + sm_tbxtb
            ind_term = it_ebxeb + it_tbxtb
        elif self.ch == 'EBxTB':
            side = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            sm_ebxeb, it_ebxeb = self.__sys_elem_abg_ebxeb__(iC[:side, :side])
            sm_tbxtb, it_tbxtb = self.__sys_elem_abg_tbxtb__(iC[side:, side:])
            sm_ebxtb, it_ebxtb = self.__sys_elem_abg_ebxtb__(iC[:side, side:])
            sm_tbxeb, it_tbxeb = self.__sys_elem_abg_tbxeb__(iC[side:, :side])
            sys_mat  = sm_ebxeb + sm_tbxtb + sm_ebxtb + sm_tbxeb
            ind_term = it_ebxeb + it_tbxtb + it_ebxtb + it_tbxeb
        # Add priors if needed
        if self.a_prior:
            for ii in range(self.Nb):
                # alpha_i
                ind_term[ii+1] += np.sum(self.a_vec * self.a_invcov[ii, :])
                for jj in range(self.Nb):
                    # alpha_i - alpha_j
                    sys_mat[ii+1, jj+1] += self.a_invcov[ii, jj]
        if self.g_prior:
            for ii in range(self.Nb):
                # gamma_i
                ind_term[ii+1+self.Nb] += np.sum(self.g_vec*self.g_invcov[ii, :])
                for jj in range(self.Nb):
                    # gamma_i - gamma_j
                    sys_mat[ii+1+self.Nb, jj+1+self.Nb] += self.g_invcov[ii, jj]
        # Solve Ax=B and calculate Fisher matrix
        ang = np.linalg.solve(sys_mat, ind_term)
        # Prefactor are chosen so that the system matrix equals the Fisher matrix
        cov = np.linalg.inv(sys_mat)
        std = np.sqrt(np.diagonal(cov))
        return ang, cov, std


    def __sys_elem_abg_ebxeb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the EBxEB contribution to the alpha+beta+gamma fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        B_ijpq     = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_BBo_ij)
        E_ijpq     = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq     = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EEo_ij)
        K_ijpq     = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TEc_ij)
        J_ijpq     = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TEc_ij)
        L_ijpq     = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_TEc_ij)
        D_ij       = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_EBo_ij)
        H_ij       = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_EBo_ij)
        tau_ij     = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_EEc_ij)
        varphi_ij  = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_BBc_ij)
        ene_ij     = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_EEc_ij)
        epsilon_ij = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_BBc_ij)
        S_ij       = self.mle.summation('ij',   self.mle.Db_TEc_ij, iC, self.mle.Db_EBo_ij)
        R_ij       = self.mle.summation('ij',   self.mle.Db_TEc_ij, iC, self.mle.Db_EEc_ij)
        T_ij       = self.mle.summation('ij',   self.mle.Db_TEc_ij, iC, self.mle.Db_BBc_ij)
        C          = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_BBc_ij)
        F          = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_EEc_ij)
        G          = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_BBc_ij)
        O          = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_EEc_ij)
        P          = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_BBc_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as beta, alpha_i, gamma_i
        sys_mat[0, 0] = 4*(G + F - 2*C)  # beta - beta
        ind_term[0]   = 2*(O - P)        # beta
        for ii in range(self.Nb):
            # beta - alpha_i
            b_a = np.sum(tau_ij[:, ii]) + np.sum(epsilon_ij[ii, :]) - np.sum(varphi_ij[:, ii]) - np.sum(ene_ij[ii, :])
            sys_mat[   0, ii+1] += 4*b_a
            sys_mat[ii+1,    0] += 4*b_a
            # beta - gamma_i
            b_g = np.sum(R_ij[:, ii]) - np.sum(T_ij[:, ii])
            sys_mat[           0, ii+1+self.Nb] += 2*b_g
            sys_mat[ii+1+self.Nb,            0] += 2*b_g
            # alpha_i
            ind_term[ii+1] += 2*(np.sum(D_ij[:, ii]) - np.sum(H_ij[ii, :]))
            # gamma_i
            ind_term[ii+1+self.Nb] += np.sum(S_ij[:, ii])
            for jj in range(self.Nb):
                # alpha_i - alpha_j terms
                aux1 = np.sum(E_ijpq[:, jj, :, ii]) + np.sum(E_ijpq[:, ii, :, jj])
                aux2 = np.sum(B_ijpq[jj, :, ii, :]) + np.sum(B_ijpq[ii, :, jj, :])
                aux3 = np.sum(I_ijpq[jj, :, :, ii]) + np.sum(I_ijpq[ii, :, :, jj])
                sys_mat[ii+1, jj+1] += 2*(aux1 + aux2 - 2*aux3)
                # alpha_i - gamma_j terms
                a_g  = np.sum(K_ijpq[:, ii, :, jj]) - np.sum(J_ijpq[ii, :, :, jj])
                sys_mat[ii+1, jj+1+self.Nb] += 2*a_g
                # gamma_i - gamma_j terms
                g_g  = np.sum(L_ijpq[:, jj, :, ii]) + np.sum(L_ijpq[:, ii, :, jj])
                sys_mat[ii+1+self.Nb, jj+1+self.Nb] += 0.5*g_g
        return sys_mat, ind_term


    def __sys_elem_abg_tbxtb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the TBxTB contribution to the alpha+beta+gamma fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TEo_ij)
        J_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TTc_ij)
        L_ijpq = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_TTc_ij)
        D_ij   = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_TBo_ij)
        tau_ij = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_TEc_ij)
        S_ij   = self.mle.summation('ij',   self.mle.Db_TTc_ij, iC, self.mle.Db_TBo_ij)
        R_ij   = self.mle.summation('ij',   self.mle.Db_TTc_ij, iC, self.mle.Db_TEc_ij)
        C      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_TEc_ij)
        O      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_TEc_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as beta, alpha_i, gamma_i
        sys_mat[0, 0] = 4*C  # beta - beta
        ind_term[0]   = 2*O  # beta
        for ii in range(self.Nb):
            # beta - alpha_i
            sys_mat[   0, ii+1] += 4*np.sum(tau_ij[:, ii])
            sys_mat[ii+1,    0] += 4*np.sum(tau_ij[:, ii])
            # beta - gamma_i
            sys_mat[           0, ii+1+self.Nb] += 2*np.sum(R_ij[:, ii])
            sys_mat[ii+1+self.Nb,            0] += 2*np.sum(R_ij[:, ii])
            # alpha_i
            ind_term[ii+1] += 2*np.sum(D_ij[:, ii])
            # gamma_i
            ind_term[ii+1+self.Nb] += np.sum(S_ij[:, ii])
            for jj in range(self.Nb):
                # alpha_i - alpha_j terms
                sys_mat[ii+1, jj+1] += 2*np.sum(I_ijpq[:, jj, :, ii])
                sys_mat[ii+1, jj+1] += 2*np.sum(I_ijpq[:, ii, :, jj])
                # alpha_i - gamma_j terms
                sys_mat[ii+1, jj+1+self.Nb] += 2 * np.sum(J_ijpq[:, ii, :, jj])
                # gamma_i - gamma_j terms
                g_g = np.sum(L_ijpq[:, jj, :, ii]) + np.sum(L_ijpq[:, ii, :, jj])
                sys_mat[ii+1+self.Nb, jj+1+self.Nb] += 0.5*g_g
        return sys_mat, ind_term


    def __sys_elem_abg_tbxeb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the TBxEB contribution to the alpha+beta+gamma fit        

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_BBo_ij)
        L_ijpq = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_TEc_ij)
        K_ijpq = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_EEo_ij)
        J_ijpq = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_BBo_ij)
        M_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TEc_ij)
        O_ij   = self.mle.summation('ij',   self.mle.Db_TTc_ij, iC, self.mle.Db_EBo_ij)
        N_ij   = self.mle.summation('ij',   self.mle.Db_TTc_ij, iC, self.mle.Db_EEc_ij)
        V_ij   = self.mle.summation('ij',   self.mle.Db_TTc_ij, iC, self.mle.Db_BBc_ij)
        P_pq   = self.mle.summation('pq',   self.mle.Db_TBo_ij, iC, self.mle.Db_TEc_ij)
        G_pq   = self.mle.summation('pq',   self.mle.Db_TEc_ij, iC, self.mle.Db_TEc_ij)
        Z_ij   = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_EBo_ij)
        W_ij   = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_EEc_ij)
        R_ij   = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_BBc_ij)
        X_pq   = self.mle.summation('pq',   self.mle.Db_TBo_ij, iC, self.mle.Db_EEo_ij)
        Y_pq   = self.mle.summation('pq',   self.mle.Db_TBo_ij, iC, self.mle.Db_BBo_ij)
        S_pq   = self.mle.summation('pq',   self.mle.Db_TEc_ij, iC, self.mle.Db_EEo_ij)
        T_pq   = self.mle.summation('pq',   self.mle.Db_TEc_ij, iC, self.mle.Db_BBo_ij)
        E      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_EEc_ij)
        B      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_BBc_ij)
        C      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_EBo_ij)
        D      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_EEc_ij)
        F      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_BBc_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as beta, alpha_i
        sys_mat[0, 0] = 4*(D - F)  # beta - beta
        ind_term[0]   = C + E - B  # beta
        for ii in range(self.Nb):
            # beta - alpha_i
            b_a = np.sum(W_ij[:, ii]) + np.sum(S_pq[:, ii]) - np.sum(R_ij[:, ii]) - np.sum(T_pq[ii, :])
            sys_mat[   0, ii+1] += 2*b_a
            sys_mat[ii+1,    0] += 2*b_a
            # beta - gamma_i
            b_g = np.sum(N_ij[:, ii]) - np.sum(V_ij[:, ii]) + np.sum(G_pq[:, ii])
            sys_mat[           0, ii+1+self.Nb] += b_g
            sys_mat[ii+1+self.Nb,            0] += b_g
            # alpha_i
            ind_term[ii+1] += np.sum(X_pq[:, ii]) + np.sum(Z_ij[:, ii]) - np.sum(Y_pq[ii, :])
            # gamma_i
            ind_term[ii+1+self.Nb] += 0.5 * (np.sum(O_ij[:, ii]) + np.sum(P_pq[:, ii]))
            for jj in range(self.Nb):
                # alpha_i - alpha_j terms
                aux1 = np.sum(H_ijpq[:, jj, :, ii]) + np.sum(H_ijpq[:, ii, :, jj])
                aux2 = np.sum(I_ijpq[:, ii, jj, :]) + np.sum(I_ijpq[:, jj, ii, :])
                sys_mat[ii+1, jj+1] += 2*(aux1 - aux2)
                # alpha_i - gamma_j terms
                sys_mat[ii+1, jj+1+self.Nb] += np.sum(K_ijpq[:, jj, :, ii]) - np.sum(J_ijpq[:, jj, ii, ::]) + np.sum(M_ijpq[:, ii, :, jj])
                # gamma_i - gamma_j terms
                g_g = np.sum(L_ijpq[:, jj, :, ii]) + np.sum(L_ijpq[:, ii, :, jj])
                sys_mat[ii+1+self.Nb, jj+1+self.Nb] += 0.5*g_g
        return sys_mat, ind_term


    def __sys_elem_abg_ebxtb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the EBxTB contribution to the alpha+beta+gamma fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TEo_ij)
        L_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_TTc_ij)
        K_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TTc_ij)
        J_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TTc_ij)
        M_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_TEo_ij)
        Z_pq   = self.mle.summation('pq',   self.mle.Db_EBo_ij, iC, self.mle.Db_TEo_ij)
        W_pq   = self.mle.summation('pq',   self.mle.Db_EEc_ij, iC, self.mle.Db_TEo_ij)
        R_pq   = self.mle.summation('pq',   self.mle.Db_BBc_ij, iC, self.mle.Db_TEo_ij)
        O_pq   = self.mle.summation('pq',   self.mle.Db_EBo_ij, iC, self.mle.Db_TTc_ij)
        N_pq   = self.mle.summation('pq',   self.mle.Db_EEc_ij, iC, self.mle.Db_TTc_ij)
        V_pq   = self.mle.summation('pq',   self.mle.Db_BBc_ij, iC, self.mle.Db_TTc_ij)
        P_ij   = self.mle.summation('ij',   self.mle.Db_TEc_ij, iC, self.mle.Db_TBo_ij)
        G_ij   = self.mle.summation('ij',   self.mle.Db_TEc_ij, iC, self.mle.Db_TEc_ij)
        X_ij   = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_TBo_ij)
        Y_ij   = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_TBo_ij)
        S_ij   = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_TEc_ij)
        T_ij   = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_TEc_ij)
        E      = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_TBo_ij)
        B      = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_TBo_ij)
        C      = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_TEc_ij)
        D      = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_TEc_ij)
        F      = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_TEc_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as beta, alpha_i, gamma_i
        sys_mat[0, 0] = 4*(D - F)  # beta - beta
        ind_term[0]   = C + E - B  # beta
        for ii in range(self.Nb):
            # beta - alpha_i
            b_a = np.sum(W_pq[:, ii]) + np.sum(S_ij[:, ii]) - np.sum(R_pq[:, ii]) - np.sum(T_ij[ii, :])
            sys_mat[   0, ii+1] += 2*b_a
            sys_mat[ii+1,    0] += 2*b_a
            # beta - gamma_i
            b_g = np.sum(N_pq[:, ii]) - np.sum(V_pq[:, ii]) + np.sum(G_ij[:, ii])
            sys_mat[           0, ii+1+self.Nb] += b_g
            sys_mat[ii+1+self.Nb,            0] += b_g
            # alpha_i
            ind_term[ii+1] += np.sum(X_ij[:, ii]) + np.sum(Z_pq[:, ii]) - np.sum(Y_ij[ii, :])
            # gamma_i
            ind_term[ii+1+self.Nb] += 0.5 * (np.sum(O_pq[:, ii]) + np.sum(P_ij[:, ii]))
            for jj in range(self.Nb):
                # alpha_i - alpha_j terms
                aux1 = np.sum(H_ijpq[:, jj, :, ii]) + np.sum(H_ijpq[:, ii, :, jj])
                aux2 = np.sum(I_ijpq[:, ii, jj, :]) + np.sum(I_ijpq[:, jj, ii, :])
                sys_mat[ii+1, jj+1] += 2*(aux1 - aux2)
                # alpha_i - gamma_j terms
                sys_mat[ii+1, jj+1+self.Nb] += np.sum(K_ijpq[:, ii, :, jj]) - np.sum(J_ijpq[ii, :, :, jj]) + np.sum(M_ijpq[:, jj, :, ii])
                # gamma_i - gamma_j terms
                g_g = np.sum(L_ijpq[:, jj, :, ii]) + np.sum(L_ijpq[:, ii, :, jj])
                sys_mat[ii+1+self.Nb, jj+1+self.Nb] += 0.5*g_g
        return sys_mat, ind_term


    #######################################################################
    ### Fit to psi
    ####################################################################### 

    def solve_pi(self, iC):
        """
        Solve linear system fitting psi. The function internally combines the 
        information from an EB-only fit, TB-only fit, fit using both EB and TB 
        but treating them as independent ('EB+TB'), and the fit including their
        cross-correlations ('EBxTB').

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        ang : np.array
            Best-fit [psi_i] values in radians
        cov : np.ndarray
            Covariance between variables
        std : np.array
            Sigma of variables
        """
        # Build the basic linear system
        if self.ch == 'EB':
            # iC is already the block you want
            sys_mat, ind_term = self.__sys_elem_pi_ebxeb__(iC)
        elif self.ch == 'TB':
            # iC is already the block you want
            sys_mat, ind_term = self.__sys_elem_pi_tbxtb__(iC)
        elif self.ch == 'EB+TB':
            side = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            sm_ebxeb, it_ebxeb = self.__sys_elem_pi_ebxeb__(iC[:side, :side])
            sm_tbxtb, it_tbxtb = self.__sys_elem_pi_tbxtb__(iC[side:, side:])
            sys_mat  = sm_ebxeb + sm_tbxtb
            ind_term = it_ebxeb + it_tbxtb
        elif self.ch == 'EBxTB':
            side = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            sm_ebxeb, it_ebxeb = self.__sys_elem_pi_ebxeb__(iC[:side, :side])
            sm_tbxtb, it_tbxtb = self.__sys_elem_pi_tbxtb__(iC[side:, side:])
            sm_ebxtb, it_ebxtb = self.__sys_elem_pi_ebxtb__(iC[:side, side:])
            sm_tbxeb, it_tbxeb = self.__sys_elem_pi_tbxeb__(iC[side:, :side])
            sys_mat  = sm_ebxeb + sm_tbxtb + sm_ebxtb + sm_tbxeb
            ind_term = it_ebxeb + it_tbxtb + it_ebxtb + it_tbxeb
        # Add priors if needed
        if self.a_prior:
            for ii in range(self.Nb):
                # psi_i
                ind_term[ii] += np.sum(self.a_vec*self.a_invcov[ii, :])
                for jj in range(self.Nb):
                    # psi_i - psi_j
                    sys_mat[ii, jj] += self.a_invcov[ii, jj]
        # Solve Ax=B and calculate Fisher matrix
        ang = np.linalg.solve(sys_mat, ind_term)
        # Prefactor are chosen so that the system matrix equals the Fisher matrix
        cov = np.linalg.inv(sys_mat)
        std = np.sqrt(np.diagonal(cov))
        return ang, cov, std


    def __sys_elem_pi_ebxeb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the EBxEB contribution to the psi fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        B_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_BBo_ij)
        E_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EEo_ij)
        D_ij   = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_EBo_ij)
        H_ij   = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_EBo_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as psi_i
        for ii in range(self.Nb):
            # psi_i
            ind_term[ii] += 2*(np.sum(D_ij[:, ii]) - np.sum(H_ij[ii, :]))
            for jj in range(self.Nb):
                # psi_i - psi_j terms
                aux1 = np.sum(E_ijpq[:, jj, :, ii]) + np.sum(E_ijpq[:, ii, :, jj])
                aux2 = np.sum(B_ijpq[jj, :, ii, :]) + np.sum(B_ijpq[ii, :, jj, :])
                aux3 = np.sum(I_ijpq[jj, :, :, ii]) + np.sum(I_ijpq[ii, :, :, jj])
                sys_mat[ii, jj] += 2*(aux1 + aux2 - 2*aux3)
        return sys_mat, ind_term


    def __sys_elem_pi_tbxtb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the TBxTB contribution to the psi fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TEo_ij)
        D_ij   = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_TBo_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as psi_i
        for ii in range(self.Nb):
            # psi_i
            ind_term[ii] += 2*np.sum(D_ij[:, ii])
            for jj in range(self.Nb):
                # psi_i - psi_j terms
                sys_mat[ii, jj] += 2*np.sum(I_ijpq[:, jj, :, ii])
                sys_mat[ii, jj] += 2*np.sum(I_ijpq[:, ii, :, jj])
        return sys_mat, ind_term


    def __sys_elem_pi_tbxeb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the TBxEB contribution to the psi fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_BBo_ij)
        Z_ij   = self.mle.summation('ij',   self.mle.Db_TEo_ij, iC, self.mle.Db_EBo_ij)
        X_pq   = self.mle.summation('pq',   self.mle.Db_TBo_ij, iC, self.mle.Db_EEo_ij)
        Y_pq   = self.mle.summation('pq',   self.mle.Db_TBo_ij, iC, self.mle.Db_BBo_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as psi_i
        for ii in range(self.Nb):
            # psi_i
            ind_term[ii] += np.sum(X_pq[:, ii]) + np.sum(Z_ij[:, ii]) - np.sum(Y_pq[ii, :])
            for jj in range(self.Nb):
                # psi_i - psi_j terms
                aux1 = np.sum(H_ijpq[:, jj, :, ii]) + np.sum(H_ijpq[:, ii, :, jj])
                aux2 = np.sum(I_ijpq[:, ii, jj, :]) + np.sum(I_ijpq[:, jj, ii, :])
                sys_mat[ii, jj] += 2*(aux1 - aux2)
        return sys_mat, ind_term


    def __sys_elem_pi_ebxtb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the EBxTB contribution to the psi fit

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.TYPE

        Returns
        -------
        sys_mat : np.ndarray
            System matrix
        ind_term : np.array
            Independent term
        """
        # Input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TEo_ij)
        Z_pq   = self.mle.summation('pq',   self.mle.Db_EBo_ij, iC, self.mle.Db_TEo_ij)
        X_ij   = self.mle.summation('ij',   self.mle.Db_EEo_ij, iC, self.mle.Db_TBo_ij)
        Y_ij   = self.mle.summation('ij',   self.mle.Db_BBo_ij, iC, self.mle.Db_TBo_ij)
        # Build system matrix and independent term
        sys_mat  = np.zeros((self.Nv, self.Nv), dtype=np.float64)
        ind_term = np.zeros(self.Nv, dtype=np.float64)
        # Variables ordered as psi_i
        for ii in range(self.Nb):
            # psi_i
            ind_term[ii] += np.sum(X_ij[:, ii]) + np.sum(Z_pq[:, ii]) - np.sum(Y_ij[ii, :])
            for jj in range(self.Nb):
                # alpha_i - alpha_j terms
                aux1 = np.sum(H_ijpq[:, jj, :, ii]) + np.sum(H_ijpq[:, ii, :, jj])
                aux2 = np.sum(I_ijpq[:, ii, jj, :]) + np.sum(I_ijpq[:, jj, ii, :])
                sys_mat[ii, jj] += 2*(aux1 - aux2)
        return sys_mat, ind_term

    #######################################################################
    ### Fit to a global rotation
    #######################################################################

    def solve_pg(self, iC):
        """
        Solve linear system fitting a global rotation. The function internally
        combines the information from an EB-only fit, TB-only fit, fit using
        both EB and TB but treating them as independent ('EB+TB'), and the fit
        including their cross-correlations ('EBxTB').

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the array-band power covariance matrix.

        Returns
        -------
        ang : float
            Best-fit psi_global in radians
        cov : float
            Variance
        std : float
            Sigma
        """
        # Build the basic linear system
        if self.ch == 'EB':
            # iC is already the block you want
            num, dem = self.__sys_elem_pg_ebxeb__(iC)
        elif self.ch == 'TB':
            # iC is already the block you want
            num, dem = self.__sys_elem_pg_tbxtb__(iC)
        elif self.ch == 'EB+TB':
            side = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            n_ebxeb, d_ebxeb = self.__sys_elem_pg_ebxeb__(iC[:side, :side])
            n_tbxtb, d_tbxtb = self.__sys_elem_pg_tbxtb__(iC[side:, side:])
            num = n_ebxeb + n_tbxtb
            dem = d_ebxeb + d_tbxtb
        elif self.ch == 'EBxTB':
            side = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            n_ebxeb, d_ebxeb = self.__sys_elem_pg_ebxeb__(iC[:side, :side])
            n_tbxtb, d_tbxtb = self.__sys_elem_pg_tbxtb__(iC[side:, side:])
            n_ebxtb, d_ebxtb = self.__sys_elem_pg_ebxtb__(iC[:side, side:])
            n_tbxeb, d_tbxeb = self.__sys_elem_pg_tbxeb__(iC[side:, :side])
            num = n_ebxeb + n_tbxtb + n_ebxtb + n_tbxeb
            dem = d_ebxeb + d_tbxtb + d_ebxtb + d_tbxeb
        if self.a_prior:
            raise NotImplementedError(r'No priors allowed when fitting $\psi_g$')
        ang = num / (8*dem)
        std = 1 / np.sqrt(4*dem)
        cov = std**2
        return ang, cov, std


    def __sys_elem_pg_ebxeb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the EBxEB contribution to the psi_global fit. In this case, they
        reduce to the numerator and denominator of a fraction.

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        float
            Numerator
        float
            Denominator
        """
        B = self.mle.summation('_', self.mle.Db_BBo_ij, iC, self.mle.Db_BBo_ij)
        E = self.mle.summation('_', self.mle.Db_EEo_ij, iC, self.mle.Db_EEo_ij)
        I = self.mle.summation('_', self.mle.Db_BBo_ij, iC, self.mle.Db_EEo_ij)
        D = self.mle.summation('_', self.mle.Db_EEo_ij, iC, self.mle.Db_EBo_ij)
        H = self.mle.summation('_', self.mle.Db_BBo_ij, iC, self.mle.Db_EBo_ij)
        return 4*(D - H), E + B - 2*I


    def __sys_elem_pg_tbxtb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the TBxTB contribution to the psi_global fit. In this case, they
        reduce to the numerator and denominator of a fraction.

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        float
            Numerator
        float
            Denominator
        """
        I = self.mle.summation('_', self.mle.Db_TEo_ij, iC, self.mle.Db_TEo_ij)
        D = self.mle.summation('_', self.mle.Db_TEo_ij, iC, self.mle.Db_TBo_ij)
        return 4*D, I


    def __sys_elem_pg_ebxtb__(self, iC):
        """
        Internal function calculating the system matrix and independent term
        from the EBxTB contribution to the psi_global fit. In this case, they
        reduce to the numerator and denominator of a fraction.        

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        float
            Numerator
        float
            Denominator
        """
        H = self.mle.summation('_', self.mle.Db_EEo_ij, iC, self.mle.Db_TEo_ij)
        I = self.mle.summation('_', self.mle.Db_BBo_ij, iC, self.mle.Db_TEo_ij)
        Z = self.mle.summation('_', self.mle.Db_EBo_ij, iC, self.mle.Db_TEo_ij)
        X = self.mle.summation('_', self.mle.Db_EEo_ij, iC, self.mle.Db_TBo_ij)
        Y = self.mle.summation('_', self.mle.Db_BBo_ij, iC, self.mle.Db_TBo_ij)
        return 2*(X + Z - Y), H - I


    def __sys_elem_pg_tbxeb__(self, iC): 
        """
        Internal function calculating the system matrix and independent term
        from the TBxEB contribution to the psi_global fit. In this case, they
        reduce to the numerator and denominator of a fraction.

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.

        Returns
        -------
        float
            Numerator
        float
            Denominator
        """
        H = self.mle.summation('_', self.mle.Db_TEo_ij, iC, self.mle.Db_EEo_ij)
        I = self.mle.summation('_', self.mle.Db_TEo_ij, iC, self.mle.Db_BBo_ij)
        Z = self.mle.summation('_', self.mle.Db_TEo_ij, iC, self.mle.Db_EBo_ij)
        X = self.mle.summation('_', self.mle.Db_TBo_ij, iC, self.mle.Db_EEo_ij)
        Y = self.mle.summation('_', self.mle.Db_TBo_ij, iC, self.mle.Db_BBo_ij)
        return 2*(X + Z - Y), H - I
