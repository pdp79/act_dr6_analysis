#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################################################
# \chi^2 associated to the Minami-Komatsu fit.
# \chi^2 computed in the small angle approximation (shouldn't be a problem).
#
# Excuse the weird naming of variables. That's the shorthand I used when 
# deriving equations on pen and paper.
#                             
# @author: Patricia Diego-Palazuelos (diegop@mpa-garching.mpg.de)
#####################################################################
import numpy as np


class MK_chi_square:
    
    def __init__(self, mle):
        """
        Object calculating the chi-square associated to the Minami-Komatsu
        estimator.

        Parameters
        ----------
        mle : MLE object
            Maximum-likelihood estimator the chi-square is associated with.

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
        self.Nv       = mle.Nvar
        self.Nb       = mle.Nbands
        #######################################################################
        ### Tricks for fast indexing
        #######################################################################
        # data type valid for <=70 bands, optimizing memory use
        self.II = np.zeros((self.Nb, self.Nb, self.Nb, self.Nb), dtype=np.uint8)
        self.JJ = np.zeros((self.Nb, self.Nb, self.Nb, self.Nb), dtype=np.uint8)
        self.PP = np.zeros((self.Nb, self.Nb, self.Nb, self.Nb), dtype=np.uint8)
        self.QQ = np.zeros((self.Nb, self.Nb, self.Nb, self.Nb), dtype=np.uint8)
        for mn_pair in mle.MNidx:
            ii, jj, pp, qq, mm, nn = mle.get_idx(mn_pair)
            self.II[ii, jj, pp, qq] = ii
            self.JJ[ii, jj, pp, qq] = jj
            self.PP[ii, jj, pp, qq] = pp
            self.QQ[ii, jj, pp, qq] = qq
            
            
            
    def dof(self):
        """
        Calculate the fit's number of degrees of freedom

        Returns
        -------
        int
            Degrees of freedom after subtracting model parameters
        """
        Npts = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
        if self.ch in ['EB+TB', 'EBxTB']:
            Npts *= 2
        return Npts - self.Nv


    #######################################################################
    ### Fit to alpha+beta
    #######################################################################  

    def calc_ab(self, angs, cov):
        """
        Calculate chi-square when fitting alpha+beta. The function internally
        combines the information from an EB-only fit, TB-only fit, fit using
        both EB and TB but treating them as independent ('EB+TB'), and the fit
        including their cross-correlations ('EBxTB').

        Parameters
        ----------
        angs : np.array
            Array containing the values of [beta, alpha_i] in radians.
        cov : np.ndarray
            Covariance matrix between cross array-band powers.

        Returns
        -------
        int
            Chi-square value.
        """
        beta, alpha = angs[0], angs[1:]
        ai, aj      = alpha[self.II], alpha[self.JJ]
        ap, aq      = alpha[self.PP], alpha[self.QQ]
        iC          = np.linalg.inv(cov)
        if self.ch == 'EB':
            chi2 = self.__chi2_ab_ebxeb__(iC, ai, aj, ap, aq, beta)
        elif self.ch == 'TB':
            chi2 = self.__chi2_ab_tbxtb__(iC, aj, aq, beta)
        elif self.ch == 'EB+TB':
            side  = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            chi2  = self.__chi2_ab_ebxeb__(iC[:side, :side], ai, aj, ap, aq, beta)
            chi2 += self.__chi2_ab_tbxtb__(iC[side:, side:], aj, aq, beta)
        elif self.ch == 'EBxTB':
            side  = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            chi2  = self.__chi2_ab_ebxeb__(iC[:side, :side], ai, aj, ap, aq, beta)
            chi2 += self.__chi2_ab_tbxtb__(iC[side:, side:], aj, aq, beta)
            chi2 += self.__chi2_ab_ebxtb__(iC[:side, side:], ai, aj, aq, beta)
            chi2 += self.__chi2_ab_tbxeb__(iC[side:, :side], aj, ap, aq, beta)
        return chi2
    

    def __chi2_ab_ebxeb__(self, iC, ai, aj, ap, aq, b):
        """
        Internal function calculating the EBxEB contribution to the total
        chi-square when fitting alpha+beta

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        ai : np.ndarray
            Matrix with alpha_i (in radians) for every cross array-band combination
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        ap : np.ndarray
            Matrix with alpha_p (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination
        b : float
            Birefringence angle in radians.

        Returns
        -------
        int
            Chi-square value.
        """
        # iC is already the block you want
        B_ijpq       = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_BBo_ij)
        E_ijpq       = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq       = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EEo_ij)
        D_ijpq       = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EBo_ij)
        H_ijpq       = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EBo_ij)
        tau_ijpq     = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EEc_ij)
        varphi_ijpq  = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_BBc_ij)
        ene_ijpq     = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EEc_ij)
        epsilon_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_BBc_ij)
        A            = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_EBo_ij)
        C            = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_BBc_ij)
        F            = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_EEc_ij)
        G            = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_BBc_ij)
        O            = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_EEc_ij)
        P            = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_BBc_ij)
        # Compute chi square
        chi2  = A + 4*b**2 * (G + F - 2*C) + 4*b * (P - O)
        chi2 += 4*np.sum(aj*aq*E_ijpq) + 4*np.sum(ai*ap * B_ijpq) - 8*np.sum(ai*aq*I_ijpq)
        chi2 += 4*np.sum(ai*H_ijpq) - 4*np.sum(aj*D_ijpq)
        chi2 += 8*b*np.sum(aj*tau_ijpq) - 8*b*np.sum(aj*varphi_ijpq)
        chi2 += 8*b*np.sum(ai*epsilon_ijpq) - 8*b*np.sum(ai*ene_ijpq)
        return chi2


    def __chi2_ab_tbxtb__(self, iC, aj, aq, b):
        """
        Internal function calculating the TBxTB contribution to the total
        chi-square when fitting alpha+beta

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination
        b : float
            Birefringence angle in radians.

        Returns
        -------
        int
            Chi-square value.
        """
        # iC is already the block you want
        I_ijpq   = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TEo_ij)
        D_ijpq   = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TBo_ij)
        tau_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TEc_ij)
        A        = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_TBo_ij)
        C        = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_TEc_ij)
        O        = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_TEc_ij)
        # Compute chi square
        chi2  = A + 4*b**2*C - 4*b*O + 4*np.sum(aj*aq*I_ijpq)
        chi2 += 8*b*np.sum(aj*tau_ijpq) - 4*np.sum(aj*D_ijpq)
        return chi2
    

    def __chi2_ab_tbxeb__(self, iC, aj, ap, aq, b):
        """
        Internal function calculating the TBxEB contribution to the total
        chi-square when fitting alpha+beta

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        ap : np.ndarray
            Matrix with alpha_p (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination
        b : float
            Birefringence angle in radians.

        Returns
        -------
        int
            Chi-square value.
        """
        # input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_BBo_ij)
        Z_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EBo_ij)
        W_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EEc_ij)
        R_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_BBc_ij)
        X_ijpq = self.mle.summation('ijpq', self.mle.Db_TBo_ij, iC, self.mle.Db_EEo_ij)
        Y_ijpq = self.mle.summation('ijpq', self.mle.Db_TBo_ij, iC, self.mle.Db_BBo_ij)
        S_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_EEo_ij)
        T_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_BBo_ij)
        A      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_EBo_ij)
        E      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_EEc_ij)
        B      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_BBc_ij)
        C      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_EBo_ij)
        D      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_EEc_ij)
        F      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_BBc_ij)
        # Compute chi square
        chi2  = A + 2*b*(B - E - C) + 4*b**2*(D - F)
        chi2 += 4*np.sum(aj*aq*H_ijpq) - 4*np.sum(aj*ap*I_ijpq)
        chi2 += 4*b*np.sum(aq*S_ijpq) + 4*b*np.sum(aj*W_ijpq)
        chi2 -= 4*b*np.sum(ap*T_ijpq) + 4*b*np.sum(aj*R_ijpq)
        chi2 += 2*np.sum(ap*Y_ijpq) - 2*np.sum(aq*X_ijpq) - 2*np.sum(aj*Z_ijpq)
        return chi2
    
    

    def __chi2_ab_ebxtb__(self, iC, ai, aj, aq, b):
        """
        Internal function calculating the EBxTB contribution to the total
        chi-square when fitting alpha+beta

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        ai : np.ndarray
            Matrix with alpha_i (in radians) for every cross array-band combination
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination
        b : float
            Birefringence angle in radians.

        Returns
        -------
        int
            Chi-square value.
        """
        # Input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TEo_ij)
        Z_ijpq = self.mle.summation('ijpq', self.mle.Db_EBo_ij, iC, self.mle.Db_TEo_ij)
        W_ijpq = self.mle.summation('ijpq', self.mle.Db_EEc_ij, iC, self.mle.Db_TEo_ij)
        R_ijpq = self.mle.summation('ijpq', self.mle.Db_BBc_ij, iC, self.mle.Db_TEo_ij)
        X_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TBo_ij)
        Y_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TBo_ij)
        S_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TEc_ij)
        T_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TEc_ij)
        A      = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_TBo_ij)
        E      = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_TBo_ij)
        B      = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_TBo_ij)
        C      = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_TEc_ij)
        D      = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_TEc_ij)
        F      = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_TEc_ij)
        # Compute chi square
        chi2  = A + 2*b*(B - E - C) + 4*b**2*(D - F)
        chi2 += 4*np.sum(aj*aq*H_ijpq) - 4*np.sum(ai*aq*I_ijpq)
        chi2 += 4*b*np.sum(aj*S_ijpq) + 4*b*np.sum(aj*W_ijpq)
        chi2 -= 4*b*np.sum(ai*T_ijpq) + 4*b*np.sum(aj*R_ijpq)
        chi2 += 2*np.sum(ai*Y_ijpq) - 2*np.sum(aj*X_ijpq) - 2*np.sum(aj*Z_ijpq)
        return chi2
    
    
    #######################################################################
    ### Fit to alpha+beta+gamma
    #######################################################################  

    def calc_abg(self, angs, cov):
        """
        Calculate chi-square when fitting alpha+beta+gamma. The function internally
        combines the information from an EB-only fit, TB-only fit, fit using
        both EB and TB but treating them as independent ('EB+TB'), and the fit
        including their cross-correlations ('EBxTB').

        Parameters
        ----------
        angs : np.array
            Array containing the values of [beta, alpha_i, gamma_i].
            Angles given in radians. 
            Gamma_i adimensional as defined in eq.(4) of the accompanying paper.
        cov : np.ndarray
            Covariance matrix between cross array-band powers.

        Returns
        -------
        int
            Chi-square value.
        """
        beta, alpha, gamma = angs[0], angs[1:-self.Nb], angs[self.Nb+1:]
        ai, aj = alpha[self.II], alpha[self.JJ]
        ap, aq = alpha[self.PP], alpha[self.QQ]
        gj, gq = gamma[self.JJ], gamma[self.QQ]
        iC     = np.linalg.inv(cov)
        if self.ch == 'EB':
            chi2 = self.__chi2_abg_ebxeb__(iC, ai, aj, ap, aq, beta, gj, gq)
        elif self.ch == 'TB':
            chi2 = self.__chi2_abg_tbxtb__(iC, aj, aq, beta, gj, gq)
        elif self.ch == 'EB+TB':
            side  = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            chi2  = self.__chi2_abg_ebxeb__(iC[:side, :side], ai, aj, ap, aq, beta, gj, gq)
            chi2 += self.__chi2_abg_tbxtb__(iC[side:, side:], aj, aq, beta, gj, gq)
        elif self.ch == 'EBxTB':
            side  = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            chi2  = self.__chi2_abg_ebxeb__(iC[:side, :side], ai, aj, ap, aq, beta, gj, gq)
            chi2 += self.__chi2_abg_tbxtb__(iC[side:, side:], aj, aq, beta, gj, gq)
            chi2 += self.__chi2_abg_ebxtb__(iC[:side, side:], ai, aj, aq, beta, gj, gq)
            chi2 += self.__chi2_abg_tbxeb__(iC[side:, :side], aj, ap, aq, beta, gj, gq)
        return chi2



    def __chi2_abg_ebxeb__(self, iC, ai, aj, ap, aq, b, gj, gq):
        """
        Internal function calculating the EBxEB contribution to the total
        chi-square when fitting alpha+beta+gamma

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        ai : np.ndarray
            Matrix with alpha_i (in radians) for every cross array-band combination
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        ap : np.ndarray
            Matrix with alpha_p (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination
        b : float
            Birefringence angle in radians.
        gj : np.ndarray
            Matrix with gamma_j (adimensional) for every cross array-band combination.
        gq : np.ndarray
            Matrix with gamma_q (adimensional) for every cross array-band combination.

        Returns
        -------
        int
            Chi-square value.
        """
        # iC is already the block you want
        B_ijpq       = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_BBo_ij)
        E_ijpq       = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq       = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EEo_ij)
        D_ijpq       = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EBo_ij)
        K_ijpq       = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TEc_ij)
        J_ijpq       = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TEc_ij)
        L_ijpq       = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_TEc_ij)
        H_ijpq       = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EBo_ij)
        tau_ijpq     = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EEc_ij)
        varphi_ijpq  = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_BBc_ij)
        ene_ijpq     = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EEc_ij)
        epsilon_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_BBc_ij)
        S_ijpq       = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_EBo_ij)
        R_ijpq       = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_EEc_ij)
        T_ijpq       = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_BBc_ij)
        A            = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_EBo_ij)
        C            = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_BBc_ij)
        F            = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_EEc_ij)
        G            = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_BBc_ij)
        O            = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_EEc_ij)
        P            = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_BBc_ij)
        # Compute chi square
        chi2  = A + 4*b**2 * (G + F - 2*C) + 4*b * (P - O)
        chi2 += 4*np.sum(aj*aq*E_ijpq) + 4*np.sum(ai*ap * B_ijpq) - 8*np.sum(ai*aq*I_ijpq)
        chi2 += 4*np.sum(ai*H_ijpq) - 4*np.sum(aj*D_ijpq)
        chi2 += 8*b*np.sum(aj*tau_ijpq) - 8*b*np.sum(aj*varphi_ijpq)
        chi2 += 8*b*np.sum(ai*epsilon_ijpq) - 8*b*np.sum(ai*ene_ijpq)
        chi2 += 4*np.sum(aj*gq*K_ijpq) - 4*np.sum(ai*gq*J_ijpq)
        chi2 += 4*b*np.sum(gq*R_ijpq) - 4*b * np.sum(gj*T_ijpq) - 2*np.sum(gj*S_ijpq)
        chi2 += np.sum(gj*gq*L_ijpq)
        return chi2

    

    def __chi2_abg_tbxtb__(self, iC, aj, aq, b, gj, gq):
        """
        Internal function calculating the TBxTB contribution to the total
        chi-square when fitting alpha+beta+gamma

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination
        b : float
            Birefringence angle in radians.
        gj : np.ndarray
            Matrix with gamma_j (adimensional) for every cross array-band combination.
        gq : np.ndarray
            Matrix with gamma_q (adimensional) for every cross array-band combination.

        Returns
        -------
        int
            Chi-square value.
        """
        # iC is already the block you want
        I_ijpq   = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TEo_ij)
        J_ijpq   = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TTc_ij)
        L_ijpq   = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_TTc_ij)
        S_ijpq   = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_TBo_ij)
        R_ijpq   = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_TEc_ij)
        D_ijpq   = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TBo_ij)
        tau_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TEc_ij)
        A        = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_TBo_ij)
        C        = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_TEc_ij)
        O        = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_TEc_ij)
        # Compute chi square
        chi2  = A + 4*b**2*C - 4*b*O + 4*np.sum(aj*aq*I_ijpq)
        chi2 += 8*b*np.sum(aj*tau_ijpq) - 4*np.sum(aj*D_ijpq)
        chi2 += 4*b*np.sum(gj*R_ijpq) - 2*np.sum(gj*S_ijpq)
        chi2 += 4*np.sum(aj*gq*J_ijpq) + np.sum(gj*gq*L_ijpq)
        return chi2
    

    def __chi2_abg_tbxeb__(self, iC, aj, ap, aq, b, gj, gq):
        """
        Internal function calculating the TBxEB contribution to the total
        chi-square when fitting alpha+beta+gamma

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination
        b : float
            Birefringence angle in radians.
        gj : np.ndarray
            Matrix with gamma_j (adimensional) for every cross array-band combination.
        gq : np.ndarray
            Matrix with gamma_q (adimensional) for every cross array-band combination.

        Returns
        -------
        int
            Chi-square value.
        """
        # input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_BBo_ij)
        Z_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EBo_ij)
        L_ijpq = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_TEc_ij)
        K_ijpq = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_EEo_ij)
        J_ijpq = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_BBo_ij)
        M_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TEc_ij)
        O_ijpq = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_EBo_ij)
        N_ijpq = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_EEc_ij)
        V_ijpq = self.mle.summation('ijpq', self.mle.Db_TTc_ij, iC, self.mle.Db_BBc_ij)
        P_ijpq = self.mle.summation('ijpq', self.mle.Db_TBo_ij, iC, self.mle.Db_TEc_ij)
        G_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_TEc_ij)
        W_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EEc_ij)
        R_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_BBc_ij)
        X_ijpq = self.mle.summation('ijpq', self.mle.Db_TBo_ij, iC, self.mle.Db_EEo_ij)
        Y_ijpq = self.mle.summation('ijpq', self.mle.Db_TBo_ij, iC, self.mle.Db_BBo_ij)
        S_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_EEo_ij)
        T_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_BBo_ij)
        A      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_EBo_ij)
        E      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_EEc_ij)
        B      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_BBc_ij)
        C      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_EBo_ij)
        D      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_EEc_ij)
        F      = self.mle.summation('_',    self.mle.Db_TEc_ij, iC, self.mle.Db_BBc_ij)
        # compute chi square
        chi2  = A + 2*b*(B - E - C) + 4*b**2*(D - F)
        chi2 += 4*np.sum(aj*aq*H_ijpq) - 4*np.sum(aj*ap*I_ijpq)
        chi2 += 4*b*np.sum(aq*S_ijpq) + 4*b*np.sum(aj*W_ijpq)
        chi2 -= 4*b*np.sum(ap*T_ijpq) + 4*b*np.sum(aj*R_ijpq)
        chi2 += 2*np.sum(ap*Y_ijpq) - 2*np.sum(aq*X_ijpq) - 2*np.sum(aj*Z_ijpq)
        chi2 += 2*np.sum(aq*gj*K_ijpq) + 2*np.sum(aj*gq*M_ijpq)
        chi2 -= np.sum(gj*O_ijpq) + 2*np.sum(ap*gj*J_ijpq) + np.sum(gq*P_ijpq)
        chi2 += 2*b*np.sum(gj*N_ijpq) - 2*b*np.sum(gj*V_ijpq) + 2*b*np.sum(gq*G_ijpq)
        chi2 += np.sum(gj*gq*L_ijpq)
        return chi2


    def __chi2_abg_ebxtb__(self, iC, ai, aj, aq, b, gj, gq):
        """
        Internal function calculating the EBxTB contribution to the total
        chi-square when fitting alpha+beta+gamma        

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        ai : np.ndarray
            Matrix with alpha_i (in radians) for every cross array-band combination
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination
        b : float
            Birefringence angle in radians.
        gj : np.ndarray
            Matrix with gamma_j (adimensional) for every cross array-band combination.
        gq : np.ndarray
            Matrix with gamma_q (adimensional) for every cross array-band combination.

        Returns
        -------
        int
            Chi-square value.
        """
        # Input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TEo_ij)
        L_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_TTc_ij)
        K_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TTc_ij)
        J_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TTc_ij)
        M_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_TEo_ij)
        Z_ijpq = self.mle.summation('ijpq', self.mle.Db_EBo_ij, iC, self.mle.Db_TEo_ij)
        W_ijpq = self.mle.summation('ijpq', self.mle.Db_EEc_ij, iC, self.mle.Db_TEo_ij)
        R_ijpq = self.mle.summation('ijpq', self.mle.Db_BBc_ij, iC, self.mle.Db_TEo_ij)
        X_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TBo_ij)
        Y_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TBo_ij)
        S_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TEc_ij)
        T_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TEc_ij)
        O_ijpq = self.mle.summation('ijpq', self.mle.Db_EBo_ij, iC, self.mle.Db_TTc_ij)
        N_ijpq = self.mle.summation('ijpq', self.mle.Db_EEc_ij, iC, self.mle.Db_TTc_ij)
        V_ijpq = self.mle.summation('ijpq', self.mle.Db_BBc_ij, iC, self.mle.Db_TTc_ij)
        P_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_TBo_ij)
        G_ijpq = self.mle.summation('ijpq', self.mle.Db_TEc_ij, iC, self.mle.Db_TEc_ij)
        A      = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_TBo_ij)
        E      = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_TBo_ij)
        B      = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_TBo_ij)
        C      = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_TEc_ij)
        D      = self.mle.summation('_',    self.mle.Db_EEc_ij, iC, self.mle.Db_TEc_ij)
        F      = self.mle.summation('_',    self.mle.Db_BBc_ij, iC, self.mle.Db_TEc_ij)
        # Compute chi square
        chi2  = A + 2*b*(B - E - C) + 4*b**2*(D - F)
        chi2 += 4*np.sum(aj*aq*H_ijpq) - 4*np.sum(ai*aq*I_ijpq)
        chi2 += 4*b*np.sum(aj*S_ijpq) + 4*b*np.sum(aj*W_ijpq)
        chi2 -= 4*b*np.sum(ai*T_ijpq) + 4*b*np.sum(aj*R_ijpq)
        chi2 += 2*np.sum(ai*Y_ijpq) - 2*np.sum(aj*X_ijpq) - 2*np.sum(aj*Z_ijpq)
        chi2 += 2*np.sum(aj*gq*K_ijpq) + 2*np.sum(aq*gj*M_ijpq)
        chi2 -= np.sum(gq*O_ijpq) + 2*np.sum(ai*gq*J_ijpq) + np.sum(gj*P_ijpq)
        chi2 += 2*b*np.sum(gq*N_ijpq) - 2*b*np.sum(gq*V_ijpq) + 2*b*np.sum(gj*G_ijpq)
        chi2 += np.sum(gj*gq*L_ijpq)
        return chi2

    #######################################################################
    ### Fit to psi
    ####################################################################### 

    def calc_pi(self, angs, cov):
        """
        Calculate chi-square when fitting psi. The function internally
        combines the information from an EB-only fit, TB-only fit, fit using
        both EB and TB but treating them as independent ('EB+TB'), and the fit
        including their cross-correlations ('EBxTB').

        Parameters
        ----------
        angs : np.array
            Array containing the values of [psi_i] in radians. 
        cov : np.ndarray
            Covariance matrix between cross array-band powers.

        Returns
        -------
        int
            Chi-square value.
        """
        ai, aj = angs[self.II], angs[self.JJ]
        ap, aq = angs[self.PP], angs[self.QQ]
        iC     = np.linalg.inv(cov)
        if self.ch == 'EB':
            chi2 = self.__chi2_pi_ebxeb__(iC, ai, aj, ap, aq)
        elif self.ch == 'TB':
            chi2 = self.__chi2_pi_tbxtb__(iC, aj, aq)
        elif self.ch == 'EB+TB':
            side  = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            chi2  = self.__chi2_pi_ebxeb__(iC[:side, :side], ai, aj, ap, aq)
            chi2 += self.__chi2_pi_tbxtb__(iC[side:, side:], aj, aq)
        elif self.ch == 'EBxTB':
            side  = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            chi2  = self.__chi2_pi_ebxeb__(iC[:side, :side], ai, aj, ap, aq)
            chi2 += self.__chi2_pi_tbxtb__(iC[side:, side:], aj, aq)
            chi2 += self.__chi2_pi_ebxtb__(iC[:side, side:], ai, aj, aq)
            chi2 += self.__chi2_pi_tbxeb__(iC[side:, :side], aj, ap, aq)
        return chi2
    

    def __chi2_pi_ebxeb__(self, iC, ai, aj, ap, aq):
        """
        Internal function calculating the EBxEB contribution to the total
        chi-square when fitting psi

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        ai : np.ndarray
            Matrix with alpha_i (in radians) for every cross array-band combination
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        ap : np.ndarray
            Matrix with alpha_p (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination

        Returns
        -------
        int
            Chi-square value.
        """
        # iC is already the block you want
        B_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_BBo_ij)
        E_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EEo_ij)
        D_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_EBo_ij)
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_EBo_ij)
        A      = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_EBo_ij)
        # compute chi square
        chi2 = A + 4*np.sum(ai*H_ijpq) - 4*np.sum(aj*D_ijpq)
        chi2 += 4*np.sum(aj*aq*E_ijpq) + 4*np.sum(ai*ap * B_ijpq) - 8*np.sum(ai*aq*I_ijpq)
        return chi2


    def __chi2_pi_tbxtb__(self, iC, aj, aq):
        """
        Internal function calculating the TBxTB contribution to the total
        chi-square when fitting psi

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination

        Returns
        -------
        int
            Chi-square value.
        """
        # iC is already the block you want
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TEo_ij)
        D_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_TBo_ij)
        A      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_TBo_ij)
        # Compute chi square
        return A + 4*np.sum(aj*aq*I_ijpq) - 4*np.sum(aj*D_ijpq)


    def __chi2_pi_tbxeb__(self, iC, aj, ap, aq):
        """
        Internal function calculating the TBxEB contribution to the total
        chi-square when fitting psi

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        ap : np.ndarray
            Matrix with alpha_p (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination

        Returns
        -------
        int
            Chi-square value.
        """
        # Input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_BBo_ij)
        Z_ijpq = self.mle.summation('ijpq', self.mle.Db_TEo_ij, iC, self.mle.Db_EBo_ij)
        X_ijpq = self.mle.summation('ijpq', self.mle.Db_TBo_ij, iC, self.mle.Db_EEo_ij)
        Y_ijpq = self.mle.summation('ijpq', self.mle.Db_TBo_ij, iC, self.mle.Db_BBo_ij)
        A      = self.mle.summation('_',    self.mle.Db_TBo_ij, iC, self.mle.Db_EBo_ij)
        # Compute chi square
        chi2  = A + 4*np.sum(aj*aq*H_ijpq) - 4*np.sum(aj*ap*I_ijpq)
        chi2 += 2*np.sum(ap*Y_ijpq) - 2*np.sum(aq*X_ijpq) - 2*np.sum(aj*Z_ijpq)
        return chi2


    def __chi2_pi_ebxtb__(self, iC, ai, aj, aq):
        """
        Internal function calculating the EBxTB contribution to the total
        chi-square when fitting alpha+beta        

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        ai : np.ndarray
            Matrix with alpha_i (in radians) for every cross array-band combination
        aj : np.ndarray
            Matrix with alpha_j (in radians) for every cross array-band combination
        aq : np.ndarray
            Matrix with alpha_q (in radians) for every cross array-band combination

        Returns
        -------
        int
            Chi-square value.
        """
        # Input iC is already the appropriate block
        H_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TEo_ij)
        I_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TEo_ij)
        Z_ijpq = self.mle.summation('ijpq', self.mle.Db_EBo_ij, iC, self.mle.Db_TEo_ij)
        X_ijpq = self.mle.summation('ijpq', self.mle.Db_EEo_ij, iC, self.mle.Db_TBo_ij)
        Y_ijpq = self.mle.summation('ijpq', self.mle.Db_BBo_ij, iC, self.mle.Db_TBo_ij)
        A      = self.mle.summation('_',    self.mle.Db_EBo_ij, iC, self.mle.Db_TBo_ij)
        # Compute chi square
        chi2  = A + 4*np.sum(aj*aq*H_ijpq) - 4*np.sum(ai*aq*I_ijpq)
        chi2 += 2*np.sum(ai*Y_ijpq) - 2*np.sum(aj*X_ijpq) - 2*np.sum(aj*Z_ijpq)
        return chi2


    #######################################################################
    ### Fit to a global rotation
    #######################################################################

    def calc_pg(self, angs, cov):
        """
        Calculate chi-square when fitting a global rotation to all bands. The 
        function internally combines the information from an EB-only fit, 
        TB-only fit, fit using both EB and TB but treating them as independent 
        ('EB+TB'), and the fit including their cross-correlations ('EBxTB').

        Parameters
        ----------
        angs : float
            Value of Psi_gobal in radians.
        cov : np.ndarray
            Covariance matrix between cross array-band powers.

        Returns
        -------
        int
            Chi-square value.
        """
        iC = np.linalg.inv(cov)
        if self.ch == 'EB':
            chi2 = self.__chi2_pg_ebxeb__(iC, angs)
        elif self.ch == 'TB':
            chi2 = self.__chi2_pg_tbxtb__(iC, angs)
        elif self.ch == 'EB+TB':
            side  = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            chi2  = self.__chi2_pg_ebxeb__(iC[:side, :side], angs)
            chi2 += self.__chi2_pg_tbxtb__(iC[side:, side:], angs)
        elif self.ch == 'EBxTB':
            side  = self.fullSize if self.bp.opt == 'fixed' else self.trimSize
            chi2  = self.__chi2_pg_ebxeb__(iC[:side, :side], angs)
            chi2 += self.__chi2_pg_tbxtb__(iC[side:, side:], angs)
            chi2 += self.__chi2_pg_ebxtb__(iC[:side, side:], angs)
            chi2 += self.__chi2_pg_tbxeb__(iC[side:, :side], angs)
        return chi2


    def __chi2_pg_ebxeb__(self, iC, a):
        """
        Internal function calculating the EBxEB contribution to the total
        chi-square when fitting a global rotation        

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        a : float
            Global rotation angle in radians.

        Returns
        -------
        int
            Chi-square value.
        """
        # iC is already the block you want
        B = self.mle.summation('_', self.mle.Db_BBo_ij, iC, self.mle.Db_BBo_ij)
        E = self.mle.summation('_', self.mle.Db_EEo_ij, iC, self.mle.Db_EEo_ij)
        I = self.mle.summation('_', self.mle.Db_BBo_ij, iC, self.mle.Db_EEo_ij)
        D = self.mle.summation('_', self.mle.Db_EEo_ij, iC, self.mle.Db_EBo_ij)
        H = self.mle.summation('_', self.mle.Db_BBo_ij, iC, self.mle.Db_EBo_ij)
        A = self.mle.summation('_', self.mle.Db_EBo_ij, iC, self.mle.Db_EBo_ij)
        # Compute chi square
        return A + 4*a*(H - D) + 4*a**2*(E + B - 2*I)


    def __chi2_pg_tbxtb__(self, iC, a):
        """
        Internal function calculating the TBxTB contribution to the total
        chi-square when fitting a global rotation   

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        a : float
            Global rotation angle in radians.

        Returns
        -------
        int
            Chi-square value.
        """
        # iC is already the block you want
        I = self.mle.summation('_', self.mle.Db_TEo_ij, iC, self.mle.Db_TEo_ij)
        D = self.mle.summation('_', self.mle.Db_TEo_ij, iC, self.mle.Db_TBo_ij)
        A = self.mle.summation('_', self.mle.Db_TBo_ij, iC, self.mle.Db_TBo_ij)
        # compute chi square
        return A + 4*a**2*I - 4*a*D


    def __chi2_pg_tbxeb__(self, iC, a):
        """
        Internal function calculating the TBxEB contribution to the total
        chi-square when fitting a global rotation   

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        a : float
            Global rotation angle in radians.

        Returns
        -------
        int
            Chi-square value.
        """
        # Input iC is already the appropriate block
        H = self.mle.summation('_', self.mle.Db_TEo_ij, iC, self.mle.Db_EEo_ij)
        I = self.mle.summation('_', self.mle.Db_TEo_ij, iC, self.mle.Db_BBo_ij)
        Z = self.mle.summation('_', self.mle.Db_TEo_ij, iC, self.mle.Db_EBo_ij)
        X = self.mle.summation('_', self.mle.Db_TBo_ij, iC, self.mle.Db_EEo_ij)
        Y = self.mle.summation('_', self.mle.Db_TBo_ij, iC, self.mle.Db_BBo_ij)
        A = self.mle.summation('_', self.mle.Db_TBo_ij, iC, self.mle.Db_EBo_ij)
        # Compute chi square
        return A + 2*a*(Y - X - Z) + 4*a**2*(H - I)


    def __chi2_pg_ebxtb__(self, iC, a):
        """
        Internal function calculating the EBxTB contribution to the total
        chi-square when fitting a global rotation   

        Parameters
        ----------
        iC : np.ndarray
            Inverse of the cross array-band power covariance matrix.
        a : float
            Global rotation angle in radians.

        Returns
        -------
        int
            Chi-square value.
        """
        # Input iC is already the appropriate block
        H = self.mle.summation('_', self.mle.Db_EEo_ij, iC, self.mle.Db_TEo_ij)
        I = self.mle.summation('_', self.mle.Db_BBo_ij, iC, self.mle.Db_TEo_ij)
        Z = self.mle.summation('_', self.mle.Db_EBo_ij, iC, self.mle.Db_TEo_ij)
        X = self.mle.summation('_', self.mle.Db_EEo_ij, iC, self.mle.Db_TBo_ij)
        Y = self.mle.summation('_', self.mle.Db_BBo_ij, iC, self.mle.Db_TBo_ij)
        A = self.mle.summation('_', self.mle.Db_EBo_ij, iC, self.mle.Db_TBo_ij)
        # Compute chi square
        return A + 2*a*(Y - X - Z) + 4*a**2*(H - I)

    

