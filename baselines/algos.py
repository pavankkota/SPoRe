"""
Baseline algorithms. Note that [1] used G=1 in baseline comparisons. Some
updates to baselines may be needed to make them compatible with G>1

Authors: Pavan Kota, Daniel LeJeune

Reference: 
[1] P. K. Kota, D. LeJeune, R. A. Drezek, and R. G. Baraniuk, "Extreme Compressed 
Sensing of Poisson Rates from Multiple Measurements," Mar. 2021.

arXiv ID:
"""
import numpy as np 
from itertools import combinations, product
from scipy.optimize import lsq_linear, minimize, LinearConstraint, Bounds
from scipy.linalg import block_diag
from abc import ABC, abstractmethod, abstractproperty
import pdb
from copy import copy
from functools import partial
from cvxopt import matrix, solvers
from scipy.stats import poisson

#from spore import mmv_models
from baselines import baselines_utils

class LinearRecoveryAlgorithm(ABC): 
    def __init__(self, pre_condition=False, pre_cond_eps=1e-10): 
        """ General linear recovery algorithm: SMV or MMV
        Parameters
        ----------
        pre_condition : boolean, optional
            If True, then if a nonnegative or nonpositive set of sensing matrices is passed into 
            self.recover(), then the algorithm will pre-condition Phi based on the method used in 
            Bruckstein et al (2008) "On the Uniqueness of Nonnegative Sparse Solutions to Underdetermined Systems of Equations."
            The default is False.
        pre_cond_eps: float
            Should be a value 0 < pre_cond_eps << 1. The default value is 1e-10
            
        Returns
        -------
        None.

        """        
        self.pre_condition = pre_condition
        self.pre_cond_eps = pre_cond_eps
        
    def rec(self, Y, Phis, pyx_func=None, group_indices=None, err_tol=None): 
        if self.pre_condition: 
            # Check if Phis are either all nonnegative or nonpositive
            if np.all(Phis >= 0) or np.all(Phis <= 0): 
                Y, Phis = self.precondition(Y, Phis)
            else: 
                raise ValueError('Pre-conditioning of this kind only recommended for entirely nonnegative or nonpositive sensing matrices')
        X = self.recover(Y, Phis, pyx_func=pyx_func, group_indices=group_indices, err_tol=err_tol)
        return X
    
    @abstractmethod
    def recover(self, Y, Phis, pyx_func=None, group_indices=None, err_tol=None): 
        pass
    
    def precondition(self, Y, Phis): 
        M,_,G = Phis.shape
        C = np.eye(M) - (1-self.pre_cond_eps)/M * np.ones((M,M))
        Phis = np.einsum('ij,jkl->ikl', C, Phis)
        Y = C @ Y       
        return Y, Phis


class BaselineSMV(LinearRecoveryAlgorithm):     
    def recover(self, Y, Phis, pyx_func=None, group_indices=None, err_tol=None):         
        if np.ndim(Y) > 1: 
            if group_indices is None: 
                _, D = Y.shape
                group_indices = np.zeros(D).astype('int')          
            X = self.recover_MMV(Y, Phis, pyx_func=pyx_func, group_indices=group_indices, err_tol=err_tol)
        else: 
            X = self.recover_SMV(Y, Phis, pyx_func=pyx_func, err_tol=err_tol)
        return X
    
    @abstractmethod
    def recover_SMV(self, y, Phi, pyx_func=None, err_tol=None): 
        """
        

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        Phi : TYPE
            DESCRIPTION.
        err_tol : 
            DESCRIPTION.
        pyx_func : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
    
        pass
    
    def recover_MMV(self, Y, Phis, pyx_func=None, group_indices=None, err_tol=None):
        """
        Solve the SMV problem for each column of Y

        Parameters
        ----------
        Y : TYPE
            DESCRIPTION.
        Phis : TYPE
            DESCRIPTION.
        err_tol : TYPE
            DESCRIPTION.
        pyx_func : TYPE, optional
            DESCRIPTION. The default is None.
        group_indices : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        M, D = Y.shape        
        _, N, G = Phis.shape
        X = np.zeros((N,D))
        if group_indices is None:
            group_indices = np.zeros(D).astype('int')

        for d in range(D):           
            X[:,d] = self.recover(Y[:,d], Phis[:,:,group_indices[d]], pyx_func=pyx_func, group_indices=group_indices, err_tol=err_tol)
            
        return X        

def cs_lsq(x, phi, y): 
    return np.linalg.norm(phi @ x - y)**2
    


def row_L0(Y, Phi, noiseTol, maxK=None, l1oracle=0): 
    """ 
    NOT used in [1] - still may need work 
    
    Optimal solver real-valued X's (non-integer constrained) 
    Solve min_X ( ||X||_{row-0} ) s.t. ||Phi X - Y||_{frob} <= noiseTol
        X is real-valued matrix        
    Try up to max K
    If multiple solutions at same sparsity within noise tolerance, pick that with least measurement error
    If none in noise tolerance through maxK, still return that with least meas. error    

    Parameters
    ----------
    maxK : int, optional
        Maximum sparsity to consider. The default is None.
    l1oracle : float [0, inf), optional
        The known sum of the signal matrix X (note X known to be >= 0). The default is 0.
        If set to 0, no l1 is used 
    Returns
    -------
    xHat : array_like
        recovered signal matrix. Shape ``(N, D)``
    """
    M, N = Phi.shape    
    _, D = Y.shape
    
    if maxK is None: 
        maxK = M-1
    
    if l1oracle < 0: 
        raise ValueError("l1oracle cannot be negative")
    elif l1oracle == 0 and maxK > (M): 
        raise ValueError("maxK must be <= M-1 if l1oracle=0 (or none is provided)")
#    elif l1oracle > 0: 
#        l1constr = LinearConstraint([ [1]*N ], l1oracle, l1oracle)
    allErrs = []
    minErr = np.inf
    for k in range(maxK): 
        # Solve for all k-column submatrices of Phi 
        subIndices = list(combinations(np.arange(N), k+1))
        #subIndices = [[3, 16]]
        
        for i in range(len(subIndices)): 
            subPhi = Phi[:, subIndices[i]]                                 
            phiStack = block_diag(*[subPhi]*D)            
            yStack = np.reshape(Y.transpose(), (D*M))
            
            if l1oracle == 0 : 
                #xTemp = lsq_linear(phiStack, yStack, bounds=(0, np.inf))
                #xTemp = lsq_linear(phiStack, yStack)
                #xTemp = xTemp.x
                #print(subPhi)
                #print(subPhi.shape)
                xTemp = np.linalg.inv(subPhi.transpose() @ subPhi) @ subPhi.transpose() @ Y            
                normErr = np.linalg.norm(subPhi @ xTemp - Y)       
                allErrs.append(normErr)
                
                xHat = np.zeros((N,D))
                xHat[subIndices[i],:] = np.reshape(xTemp, (len(subIndices[i]), D))    
                #x0 = np.ones((subPhi.shape[1], D)) / l1oracle 
                #xTemp = minimize(cs_lsq, x0, args=(subPhi, Y), bounds=(np.zeros(np.shape(x0)), np.ones(np.shape(x0))*np.inf))                                                
                #normErr = np.linalg.norm(phiStack @ xTemp - yStack)        
            else: 
                #raise ValueError("l1oracle not coded yet")
                #x0 = np.ones((subPhi.shape[1], D)) / l1oracle 
                #xTemp = minimize(cs_lsq, x0, args=(subPhi, Y), bounds=(0, np.inf), constraints=(l1constr))                
                
                dim = D*subPhi.shape[1]
                
                
                #Slow 
                l1constr = LinearConstraint([ [1]*dim ], l1oracle, l1oracle)
                x0 = np.ones(dim)/l1oracle
                
                B = Bounds(0, np.inf)
                xTemp = minimize(cs_lsq, x0, args=(phiStack, yStack), bounds=B, constraints=(l1constr))                
                xTemp = xTemp.x
                #normErr = np.linalg.norm(subPhi@Y - xTemp)
                normErr = np.linalg.norm(phiStack@xTemp - yStack)
                
                xHat = np.zeros((N,D))
                xHat[subIndices[i],:] = np.transpose(np.reshape(xTemp, (D, len(subIndices[i]))))
                """
                #cvx opt
                PHI = matrix(phiStack)
                y = matrix(yStack)
                #G = matrix(np.ones(dim))
                
                A = matrix([1.0]*dim, (1, dim))                
                b = matrix(l1oracle)
                x = solvers.coneqp(PHI.T*PHI, -PHI.T*y, A=A, b=b)
                #x = solvers.coneqp(A.T*A, -A.T*b, G, h, dims)['x']                
                xTemp = np.array(x['x'])
                normErr = np.linalg.norm(phiStack@xTemp - yStack)
                """
                #pdb.set_trace()
                allErrs.append(normErr)

            if normErr < minErr:                 
                minErr = normErr
                xOut = xHat
                #xHat = np.zeros((N, D))                
                #xHat[subIndices[i], :] =     
                #xHat[subIndices[i],:] = 
                
            #print('Completed: k'+str(k+1) + '; subPhi ' + str(i+1) + '/' + str(len(subIndices)))
        if minErr < noiseTol:             
            break    
        
            
    return xOut, allErrs 
        

class oracle_int_MMV(LinearRecoveryAlgorithm): 
    """ L0-Oracle used in [1] 
    """
    def __init__(self, l0, maxX, pre_condition=False, pre_cond_eps=1e-10): 
        self.pre_condition = pre_condition
        self.pre_cond_eps = pre_cond_eps
        self.l0 = l0
        self.maxX = maxX
                    
    def recover(self, Y, Phis, pyx_func=None, group_indices=None, err_tol=None):     
        M, D = np.shape(Y)        
        _, N, G = np.shape(Phis)        
        if group_indices is None:
            group_indices = np.zeros(D).astype('int')
                
        # All N choose k combinations of atoms
        subIndices = list(combinations(np.arange(N), self.l0))        
        
        # Generate array of all x's to test
        x_i = [np.arange(self.maxX+1)]*self.l0
        xTest = np.array(list(product(*x_i))).T # self.l0 x ((self.l0)^(self.l0)) integer array
        
        minErr = np.inf
        for i in range(len(subIndices)): 
            subPhis = Phis[:, subIndices[i],:]   
            xSol = np.zeros((self.l0,D))
            solErr = 0
            for g in range(G): 
                yTest = subPhis[:,:,g] @ xTest # M, ... array
                errs = Y[:,None,group_indices==g] - yTest[:,:,None]
                normErr = np.linalg.norm(errs, axis=0)
                inds = np.argmin(normErr, axis=0)          
                xSol[:,group_indices==g] = xTest[:, inds]
                solErr += np.linalg.norm(Y[:,group_indices==g] - subPhis[:,:,g] @ xSol[:,group_indices==g])
            if solErr < minErr:           
                minErr = solErr                                 
                X = np.zeros((N,D))
                X[subIndices[i],:] = xSol

        return X
    
    
class SOMP(LinearRecoveryAlgorithm): 
    """Not used in [1] 
        Simultaenous Orthogonal Matching Pursuit (S-OMP) [2] or OMP for Multiple-
        Measurement Vector (OMPMMV) [3]. Equivalent algorithms developed simultaneously
        [2] J Tropp et al. "Algorithms for simultaneous sparse approximation: Part I" 
        [3] J Chen and X Huo. "Theoretical results on sparse representations of multiple-measurement vectors"
    """
    def __init__(self, pre_condition=False, pre_cond_eps=1e-10, k_known=None): 
        self.pre_condition = pre_condition
        self.pre_cond_eps = pre_cond_eps
        self.k_known = k_known
        
    def recover(self, Y, Phis, pyx_func=None, group_indices=None, err_tol=None): 
        """ 
        Only works if G = 1
        Parameters
        ----------
        Y : numpy array
            Measurement matrix (M x D)
        Phi : numpy array  
            Sensing matrix (M x N)
        noiseTol : TYPE
            DESCRIPTION.
    
        Returns
        -------
        Xhat : TYPE
            DESCRIPTION.
    
        """
        M, D = Y.shape
        _, N, G = Phis.shape        
        if G > 1: 
            raise ValueError('G must be 1 for SOMP')
        if err_tol is None: 
            err_tol = -np.inf # go until maxSupp elements have been picked
        
        if self.k_known is None:            
            max_supp = M # can't meaningfully go higher than M
        else: 
            max_supp = self.k_known
            
        Phi = Phis[:,:,0]
        normPhi = Phi / np.linalg.norm(Phi, axis=0)
        R = Y 
        supp = []        
        subPhi = np.zeros( (M,0) ) 
        xHat = np.zeros((N, D))    
        while np.linalg.norm(R) > err_tol or len(supp) <= max_supp: 
            # Pick column that correlates most with residuals
            newSupp = np.argmax(np.sum(np.absolute(normPhi.transpose() @ R, axis=1)))
            if len(supp.append(newSupp)) != len(set(supp.append(newSupp))): 
                break    
            supp.append(newSupp)
            subPhi = np.concatenate( (subPhi, Phi[:, newSupp, None]), axis=1)
                
            #Solve least squares min_X ( ||Y-subPhi@X||_F ) and reset residuals
            xTemp = np.linalg.inv(subPhi.transpose() @ subPhi) @ subPhi.transpose() @ Y            
            R = Y - subPhi @ xTemp                    
        
        lamHat = np.sum(xHat, axis=1)/D 
        return xHat, lamHat


class DCS_SOMP(LinearRecoveryAlgorithm): 
    """ Used in [1]
    Generalization of SOMP for G>1
    """
    def __init__(self, pre_condition=False, pre_cond_eps=1e-10, k_known=None): 
        self.pre_condition = pre_condition
        self.pre_cond_eps = pre_cond_eps
        self.k_known = k_known      
    
    def recover(self, Y, Phis, pyx_func=None, group_indices=None, err_tol=None):         
        M, D = np.shape(Y)        
        _, N, G = np.shape(Phis)
        
        if group_indices is None:
            group_indices = np.zeros(D).astype('int')
        if err_tol is None: 
            err_tol = -np.inf # go until maxSupp elements have been picked
        
        if self.k_known is None:            
            max_supp = M # can't meaningfully go higher than M
        else: 
            max_supp = self.k_known
            
        R = copy(Y)
        S = []
        Gammas = []        
        counter = 0
        xTemp = np.zeros((M,D))
        
        while np.linalg.norm(R) > err_tol and (len(S) < max_supp and len(S) < M): # in case M < self.k_known, then hitting M selections should stop too
            newSupp = self.pick_index(R, Phis, group_indices)    
            if newSupp in S: 
                print('Warning: DCS-SOMP picked the same index twice. Residual likely ~= 0, check this. Breaking loop early')
                break
            S.append(newSupp)
            for g in range(G): 
                # Orthogonalize: 
                if counter == 0: 
                    Gammas.append(Phis[:,S,g])
                else:                 
                    phiColSelect = Phis[:,S[-1],g]
                    gammaNew = phiColSelect - np.sum((phiColSelect.T @ Gammas[g]) / (np.linalg.norm(Gammas[g], axis=0)**2) * Gammas[g], axis=1)
                    Gammas[g] = np.concatenate((Gammas[g], gammaNew[:,None]), axis=1)                

                # Iterate: update coefficients and re-solve residuals: 
                xTemp[counter,group_indices==g] = R[:,group_indices==g].T @ Gammas[g][:,counter] / ((np.linalg.norm(Gammas[g][:,counter]))**2)
                
                R[:,group_indices==g] = R[:, group_indices==g] - (R[:,group_indices==g].T @ Gammas[g][:,counter] / ((np.linalg.norm(Gammas[g][:,counter]))**2)) * Gammas[g][:,counter][:,None]
                
            counter += 1
        
        # De-orthogonalize
        X = np.zeros((N,D))      
        for g in range(G): 
            Rg = np.linalg.pinv(Gammas[g]) @ Phis[:,S,g]
            xMut = np.linalg.inv(Rg) @ xTemp[:counter,group_indices==g]
            X[np.asarray(S)[:,None], group_indices==g] = xMut            

        return X
    
    def pick_index(self,R,Phis,group_indices): 
        _, N, G = Phis.shape
        scores = np.zeros(N)
        for g in range(G):                 
            scores += (np.sum(np.absolute(Phis[:,:,g].T @ R), axis=1)) / (np.linalg.norm(Phis[:,:,g], axis=0))        
        newSupp = np.argmax(scores)

        return newSupp
    

class NNS(LinearRecoveryAlgorithm): 
    """ Used in [1]
    Nonnegative Subspace Pursuit 
    D. Kim and J. P. Haldar, “Greedy algorithms for nonnegativity-constrained simultaneous sparse recovery,” Signal Processing, vol. 125, pp. 274–289, 2016.
    
    Lightly modified to incorporate groups of signals (generalized NNS-SP and NNS-CoSaMP algorithms)
    """    
    def __init__(self, k_known, pre_condition=False, pre_cond_eps=1e-10): 
        self.pre_condition = pre_condition
        self.pre_cond_eps = pre_cond_eps
        self.k_known = k_known
            
    def recover(self, Y, Phis, pyx_func=None, group_indices=None, err_tol=None):   
        M, D = np.shape(Y)        
        _, N, G = np.shape(Phis)        
        if group_indices is None:
            group_indices = np.zeros(D).astype('int')
        if err_tol is None: 
            err_tol = -np.inf # go until maxSupp elements have been picked
            
        R = copy(Y)
        Supp = np.array([])
        
        resNormOld = np.inf
        xHatOld = np.zeros((N,D))
        while 1: 
            Supp = np.union1d(Supp, self.step1(R, Phis, group_indices))
            xHat = self.x_supp_constrained(R, Phis, group_indices, Supp)
            Supp = np.argpartition(np.linalg.norm(xHat, axis=1), -self.k_known)[-self.k_known:] 
            xHat = self.step5(R, Phis, group_indices, Supp, xHat)
            
            resNorm = 0
            for g in range(G): 
                R[:,group_indices==g] = Y[:,group_indices==g] -Phis[:,:,g] @ xHat[:,group_indices==g]
                resNorm += np.linalg.norm(R[:,group_indices==g]
                                          )            
            # Stop if solution worsens or stagnates
            if resNorm >= resNormOld: 
                X = xHatOld
                break
            else: 
                resNormOld = resNorm
                xHatOld = copy(xHat)            

        return X
    
    def pick_indices(self, R, Phis, group_indices, S): 
         _, N, G = Phis.shape
         scores = np.zeros(N)
         for g in range(G):             
             # in paper, they don't divide by norm explicitly, but they note that they assume phi has unit norm columns
             mu_nn = (Phis[:,:,g].T @ R[:,group_indices==g]) / (np.linalg.norm(Phis[:,:,g], axis=0))[:,None]  
             mu_nn[mu_nn < 0] = 0
             scores += np.sum(mu_nn,axis=1)

         inds = np.argpartition(scores, -S)[-S:]    
         return inds
     
    def x_supp_constrained(self, R, Phis, group_indices, Supp):      
        M, N, G = np.shape(Phis)
        _, D = np.shape(R)            
        xHat = np.zeros((N,D))
        s = Supp.astype(int)
        
        """
        for g in range(G):             
            subPhi = Phis[:,:,g][:, s]                                 
            phiStack = block_diag(*[subPhi]*(np.sum(group_indices==g)))            
            yStack = np.reshape(R[:,group_indices==g].transpose(), (M*np.sum(group_indices==g)))                                                
            

            xTemp = lsq_linear(phiStack, yStack, bounds=(0, np.inf))['x']
            xHat[s[:,None],group_indices==g] = np.reshape(xTemp, (np.size(Supp), np.sum(group_indices==g)))    
        """  
        for d in range(D): 
            subPhi = Phis[:,s,group_indices[d]]
            #xHat[s,d] = lsq_linear(subPhi, R[:,d], bounds=(0,np.inf), method='bvls', tol=1e-6)['x']
            xHat[s,d] = lsq_linear(subPhi, R[:,d], bounds=(0,np.inf))['x']
        return xHat
    
   
    @abstractmethod
    def step1(self, R, Phis, group_indices, S): 
        pass

    @abstractmethod
    def step5(self, xHat, R, Phis, group_indices): 
        pass
    
class NNS_SP(NNS): 
    def step1(self, R, Phis, group_indices):
        return self.pick_indices(R, Phis, group_indices, self.k_known)        
    def step5(self, R, Phis, group_indices, Supp, X): 
        return self.x_supp_constrained(R, Phis, group_indices, Supp)        

class NNS_CoSaMP(NNS): 
    def step1(self, R, Phis, group_indices):
        return self.pick_indices(R, Phis, group_indices, 2*self.k_known)        
    def step5(self, R, Phis, group_indices, Supp, X): 
        _, N, _ = Phis.shape
        X[np.setdiff1d(np.arange(N), Supp), :] = 0
        return self.x_supp_constrained(R, Phis, group_indices, Supp)        
                
                
# ---- SMV baselines ----
class PROMP(BaselineSMV): 
    """
    used in [1]
    """
    def __init__(self, theta, pos_constrained=False, k_known=None, pre_condition=False, pre_cond_eps=1e-10): 
        self.pre_condition = pre_condition
        self.pre_cond_eps = pre_cond_eps 
        self.theta = theta
        self.pos_constrained = pos_constrained
        self.k_known = k_known        
    
    """
    def recover(self, Y, Phis, err_tol, pyx_func=None, group_indices=None):         
        if np.ndim(Y) > 1: 
            if group_indices is None: 
                _, D = Y.shape
                group_indices = np.zeros(D)                
            X = self.recover_MMV(Y, Phis, err_tol, pyx_func=pyx_func, group_indices=group_indices)
        else: 
            X = self.recover_SMV(Y, Phis, err_tol, pyx_func=pyx_func)
        return X
    """
    
    def recover_SMV(self, y, Phi, pyx_func=None, err_tol=None):
        M, N = Phi.shape                
        normPhi = Phi / np.linalg.norm(Phi, axis=0)        
                
        x_leastnorm = Phi.T @ np.linalg.inv((Phi @ Phi.T)) @ y 
        
        if self.pos_constrained is False: 
            S = np.where( (np.absolute(x_leastnorm) * N/M) > self.theta)[0]
        else: 
            S = np.where( (x_leastnorm * N/M) > self.theta)[0]
    
        if err_tol is None: 
            err_tol = -np.inf # go until maxSupp elements have been picked
        
        if self.k_known is None:            
            max_supp = M # can't meaningfully go higher than M
        else: 
            max_supp = self.k_known
            
        subPhi = Phi[:, S]    
        
        if np.size(S) >= M: # not overdetermined - get least norm solution with no error            
            xTemp = subPhi.T @ np.linalg.inv((subPhi @ subPhi.T)) @ y 
        else: # least squares solution
            xTemp = np.linalg.inv(subPhi.T @ subPhi) @ subPhi.T @ y 
    
        ro = y - subPhi @ xTemp
    
        while np.linalg.norm(ro) > err_tol and (np.size(S) < max_supp and np.size(S) < M): # in case M < self.k_known, then hitting M selections should stop too
            corrCheck = np.absolute(normPhi.T @ ro) 
            if np.argmax(corrCheck) in S: # shouldn't happen                
                print('Warning: PROMP picked same element again - breaking out of loop')
                break
            newSupp = np.argmax(corrCheck)
                    
            S = np.concatenate((S, np.array([newSupp])))    
            subPhi = Phi[:,S]
    
            xTemp = np.linalg.inv(subPhi.T @ subPhi) @ subPhi.T @ y 
            ro = y - subPhi @ xTemp
            
        x = np.zeros(N,)
        x[S] = np.round(xTemp)
            
        return x

class SumPoissonSMV(BaselineSMV):       
    """
    Used in [1].
    Solve by branch and bound (B&B) based on an input lambda_total (sum_n lambda_n estimate)    
    """
    # Need to figure out how to handle Sigma_inv 

    def __init__(self, Sigma_inv, lambda_total=None, alpha=1e-7, nu=0.9, poiss_cdf=1-(1e-3), epsilon=1e-6, max_iter_bb=1e4, conv_rel=1e-3):
        # No need for preconditioning on this - convex, integer solution
        
        if lambda_total is not None: 
            self.lambda_total = lambda_total       
            #TO DO Need to handle the 'none' option - estimate lambda_total from data and pyx_func
            
        self.Sigma_inv = Sigma_inv
        self.alpha = alpha
        self.nu = nu
        self.poiss_cdf = poiss_cdf
        self.epsilon = 1e-6
        self.max_iter_bb = max_iter_bb
        self.conv_rel = conv_rel
        
    def rec(self, Y, Phis, pyx_func=None, group_indices=None, xTrue=None):         
        return self.recover(Y, Phis, xTrue, pyx_func=pyx_func, group_indices=group_indices)
    
    def recover(self, Y, Phis, xTrue, pyx_func=None, group_indices=None):         
        if np.ndim(Y) > 1: 
            M, D = Y.shape
            if group_indices is None:                 
                group_indices = np.zeros(D).astype('int')
                                
            X = self.recover_MMV(Y, Phis, xTrue, pyx_func=pyx_func, group_indices=group_indices)
        else: 
            X = self.recover_SMV(Y, Phis, pyx_func=pyx_func)
        return X
    
    def recover_SMV(self, y, Phi, err_tol=None, pyx_func=None):        
        pass
        #return x
    
    
    def recover_MMV(self, Y, Phis, xTrue, pyx_func=None, group_indices=None):
        """
        Solve the SMV problem for each column of Y
        """
        M, D = Y.shape        
        _, N, G = Phis.shape
        X = np.zeros((N,D))
        if group_indices is None:
            group_indices = np.zeros(D).astype('int')
        if xTrue is None: 
            xT = None
        
        maxX = -1
        p = 0
        while p < self.poiss_cdf: 
            maxX+=1
            p = poisson.cdf(maxX, self.lambda_total)
            
        x0 = leastnorm_LSQ(Y, Phis, group_indices)
        
        lossFlags = 0
        exitFlags = 0
        for d in range(D):    
            if xTrue is not None: 
                xT = xTrue[:,d][:,None]
                    
            gdmodel = baselines_utils.GaussianAndSumPoissonLoss(Y[:,d][:,None], Phis[:,:,group_indices[d]][:,:,None], \
                                                                self.Sigma_inv[:,:,group_indices[d]][:,:,None], self.lambda_total, \
                                                                    group_indices, epsilon=self.epsilon)            
            optimizer = partial(baselines_utils.grad_descent_backtracking, gdmodel.loss, gdmodel.grad, self.alpha, xTrue=xT)
            bb = baselines_utils.BranchAndBoundMostFractional(optimizer, gdmodel.loss)
            xhat, exitFlag = bb.solve(x0[:,d][:,None], np.zeros((N,1)), np.ones((N,1))*maxX, max_iter = self.max_iter_bb)
            X[:,d] = xhat[:,0]
            print('Signal ' + str(d+1) + '/' + str(D) + 'completed')
            if exitFlag is True: 
                print('B&B reached max iterations and quit early')
                exitFlags+=1
            if xTrue is not None: 
                if gdmodel.loss(xhat) > gdmodel.loss(xT): 
                    print('Incorrect solution found: solution loss: ' + str(gdmodel.loss(xhat)) + '; true loss: ' + str(gdmodel.loss(xT)))
                    # This appears nearly inevitable to happen *occasionally* with variable simulation conditions.
                    # No consistent hyperparameter choice will avoid this always, but user
                    # should check if it's happening often
                    lossFlags += 1
                    
        return X, exitFlags, lossFlags
    
    
class PoissonAlt(BaselineSMV):        
    # Need to figure out how to handle Sigma_inv 
    """Used in [1]
    Given input of lambda, estimate X one vector at a time by branch and bound (B&B)
    """
    def __init__(self, Sigma_inv, max_alt=10, alpha=1e-7, nu=0.9, poiss_cdf=1-(1e-3), epsilon=1e-6, max_iter_bb=1e4, conv_rel=1e-3):   
        self.Sigma_inv = Sigma_inv
        self.max_alt = max_alt
        self.alpha = alpha
        self.nu = nu
        self.poiss_cdf = poiss_cdf
        self.epsilon = 1e-6
        self.max_iter_bb = max_iter_bb
        self.conv_rel=conv_rel
        
    def rec(self, Y, Phis, lam0, pyx_func=None, group_indices=None, xTrue=None):         
        return self.recover(Y, Phis, lam0, xTrue, pyx_func=pyx_func, group_indices=group_indices)
    
    def recover(self, Y, Phis, lam0, xTrue, pyx_func=None, group_indices=None):         
        if np.ndim(Y) > 1: 
            M, D = Y.shape
            if group_indices is None:                 
                group_indices = np.zeros(D).astype('int')
                                
            X = self.recover_MMV(Y, Phis, lam0, xTrue, pyx_func=pyx_func, group_indices=group_indices)
        else: 
            X = self.recover_SMV(Y, Phis, lam0, pyx_func=pyx_func)
        return X
    
    def recover_SMV(self, y, Phi, err_tol=None, pyx_func=None):        
        pass
        #return x
    
    
    def recover_MMV(self, Y, Phis, lam0, xTrue, pyx_func=None, group_indices=None):
        """
        Solve the SMV problem for each column of Y
        """
        M, D = Y.shape        
        _, N, G = Phis.shape
        X = np.zeros((N,D))
        if group_indices is None:
            group_indices = np.zeros(D).astype('int')
        if xTrue is None: 
            xT = None
            
        lamHat = lam0    

        altCount = 1
        
        lossFlags = 0
        exitFlags = 0
        while altCount <= self.max_alt: 
            if altCount == 1:
                x0 = leastnorm_LSQ(Y, Phis, group_indices)
                #x0 = np.zeros((N,D))
            else: 
                x0 = copy(X)
                
            maxX = np.zeros((N,1))        
            for n in range(N): 
                p = 0
                while p < self.poiss_cdf: 
                    maxX[n] += 1
                    p = poisson.cdf(maxX[n], lam0[n])

            for d in range(D):      
                gdmodel = baselines_utils.GaussianAndPoissonLoss(Y[:,d][:,None], Phis[:,:,group_indices[d]][:,:,None], \
                                                                    self.Sigma_inv[:,:,group_indices[d]][:,:,None], lamHat, \
                                                                        group_indices, epsilon=self.epsilon)

                if xTrue is not None: 
                    xT = xTrue[:,d][:,None]
                #optimizer = partial(baselines_utils.grad_descent, gdmodel.loss, gdmodel.grad, self.alpha, nu=self.nu, xTrue=xT, conv_rel=self.conv_rel)
                optimizer = partial(baselines_utils.grad_descent_backtracking, gdmodel.loss, gdmodel.grad, self.alpha, xTrue=xT)
                bb = baselines_utils.BranchAndBoundMostFractional(optimizer, gdmodel.loss)
                #xhat = bb.solve(x0[:,d][:,None], np.zeros((N,1)), np.ones((N,1))*maxX, max_iter = self.max_iter_bb)
                xhat, exitFlag = bb.solve(x0[:,d][:,None], np.zeros((N,1)), maxX, max_iter = self.max_iter_bb)
                X[:,d] = xhat[:,0]                
                print('Alt #' + str(altCount) + '; Signal ' + str(d+1) + '/' + str(D) + 'completed')                        
                if exitFlag is True: 
                    print('B&B reached max iterations and quit early')
                    exitFlags+=1
                if xTrue is not None:                 
                    if gdmodel.loss(xhat) > gdmodel.loss(xT): 
                        print('Incorrect solution found: solution loss: ' + str(gdmodel.loss(xhat)) + '; true loss: ' + str(gdmodel.loss(xT)))
                        lossFlags += 1
                        
            print('Alternation ' + str(altCount+1) + 'completed')                        

            lamHatNew = np.sum(X, axis=1)/D # update lambda estimate         
            if np.all(lamHatNew == lamHat): 
                break
            else: 
                lamHat = lamHatNew
                            
            altCount+=1

        return X, exitFlags, lossFlags
    
def leastnorm_LSQ(Y, Phis, group_indices=None): 
    M,N,G = Phis.shape
    _,D = Y.shape
    if group_indices is None: 
        group_indices=np.zeros(D)
    
    X = np.zeros((N,D))
    for g in range(G): 
        X[:,group_indices==g] = Phis[:,:,g].T @ np.linalg.inv((Phis[:,:,g] @ Phis[:,:,g].T)) @ Y[:,group_indices==g]        
    return X 

def x_eval(Y, Phis, X, group_indices=None):
    M,N,G = Phis.shape
    _,D = Y.shape
    if group_indices is None: 
        group_indices=np.zeros(D)
    
    err = np.zeros((M,D))
    for g in range(G): 
        err[:,group_indices==g] = Phis[:,:,g] @ X[:,group_indices==g] - Y[:,group_indices==g]
    
    return np.linalg.norm(err)