# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:56:25 2020

Authors: Pavan Kota, Daniel LeJeune

Reference: 
P. K. Kota, D. LeJeune, R. A. Drezek, and R. G. Baraniuk, "Extreme Compressed 
Sensing of Poisson Rates from Multiple Measurements," Mar. 2021.

arXiv ID:
"""

# Multiple Measurement Vector Compressed Sensing

from abc import ABC, abstractmethod  
import numpy as np 
import pickle

class SignalGenerator(ABC):
    """Methods for generating X 
    """
    @abstractmethod 
    def xgen(self, N, D, k): 
        """Generate an N x D signal matrix X 
        Parameters
        ----------
        N: int
            Dimension of signals
        D: int
            Number of N-dimensional signals to generate
        k: int
            Sparsity level. Number of nonzero elements in lambda^* (true Poisson rates)
        Returns
        -------
        X : (N, D) ndarray
            Samples of X for each column of Y.
        
        """
        pass
    
    
class MMVP(SignalGenerator):
    """ Multiple Measurement Vector with Poisson constraints (MMVP) signal generator
    """
    def __init__(self, N, D, k, lamTot, initialSeed=None): 
        """
        New Parameters
        ----------
        lamTot: float or int
            Sum(lambda^*). Corresponds with, for example, average analyte number per observation
        initialSeed: int, optional
            Seed for restoring RNG if X's are generated multiple times in same 
            script and generating the initial X's again is desired. 
        """
        if k > N :
            raise ValueError("k must be less than N")
        self.N = N 
        self.D = D
        self.k = k
        self.lamTot = lamTot     
        self.initialSeed = initialSeed 
        #np.random.seed(initialSeed) 
        self._generator = np.random.default_rng(initialSeed)
        
    def set_lambda(self): 
        lambdaStar = np.zeros(self.N)
        # Choose sparse rows randomly               
        rowInds = np.random.choice(self.N, self.k, replace=False)
        
        # Set lambda randomly 
        lambdaStar[rowInds] = self.get_mags()   
        return lambdaStar
    
    def xgen(self): 
        lambdaStar = self.set_lambda()        
        # Generate X's        
        X = self._generator.poisson(lambdaStar[:, None], (self.N, self.D))    
        
        return X, lambdaStar
    
    def gen_trials(self, numTrials, seed=None, savePath=None):
        """
        Parameters
        ----------
        numTrials : int
            Number of trials to generate sensing matrices for
        seed : int, optional
            Random seed initial state. The default is None.
        savePath: string or None
            Path including filename (.pickle file type) to store generated
            X's and lambda^*'s. If None, signals are not saved.
        """
        
        # Which to use? Need consistent selection of k rows too     
        if seed is None: 
            np.random.seed(self.initialSeed) 
            self._generator = np.random.default_rng(self.initialSeed) 
        else:
            np.random.seed(seed)
            self._generator = np.random.default_rng(seed)                        
        
        allX = np.zeros((self.N, self.D, numTrials))
        allLambdaStars = np.zeros((self.N, numTrials))
        
        for i in range(numTrials): 
            allX[:,:,i], allLambdaStars[:,i] = self.xgen()
        
        
        if savePath is not None:
            allSignals = {'signalModelUsed': self, 'allX': allX, 'allLambdaStars': allLambdaStars}
            with open(savePath, 'wb') as fileWrite:
                pickle.dump(allSignals, fileWrite)
                
        return allX, allLambdaStars
    def get_mags(self):     
        mags = self._generator.uniform(size=self.k)
        return mags / np.sum(mags) * self.lamTot        

class MMVPConstantLambda(MMVP): 
    def __init__(self, N, D, k, lambda_val, initialSeed=None): 
        """
        New Parameters
        ----------
        lambda_val: float or int
            Value to set any nonzero value of lambda to
        """
        if k > N :
            raise ValueError("k must be less than N")
        self.N = N 
        self.D = D
        self.k = k
        self.lambda_val = lambda_val
        self.initialSeed = initialSeed 
        self._generator = np.random.default_rng(initialSeed)
    def get_mags(self):     
        return np.ones(self.k) * self.lambda_val        



class MMVPInputLambda(MMVP): 
    def __init__(self, D, lambda_vec, initialSeed=None): 
        """
        New Parameters
        ----------
        lambda_vec: numpy array, shape (N,)
           Fixed lambda vector 
        """
        self.lam = lambda_vec
        self.N = np.size(lambda_vec)
        self.D = D     
        self.initialSeed = initialSeed 
        self._generator = np.random.default_rng(initialSeed)
    
    
    def set_lambda(self): 
        return self.lam
        
    def get_mags(self): 
        pass
    