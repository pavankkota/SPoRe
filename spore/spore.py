"""
Sparse Poisson Recovery (SPoRe) module for solving Multiple Measurement Vector 
problem with Poisson signals (MMVP) by batch stochastic gradient ascent and 
Monte Carlo integration

Authors: Pavan Kota, Daniel LeJeune

Reference: 
[1] P. K. Kota, D. LeJeune, R. A. Drezek, and R. G. Baraniuk, "Extreme Compressed 
Sensing of Poisson Rates from Multiple Measurements," Mar. 2021.

arXiv ID:
    
"""


from abc import ABC, abstractmethod

import numpy as np
import time
import pdb

from .mmv_models import FwdModelGroup, SPoReFwdModelGroup

class SPoRe(object):
    def __init__(self, N, fwdmodel, sampler, batch_size=100, step_size=1e-1,
        min_lambda=1e-3, pyx_min=0, grad_scale=5e-2, conv_rel=1e-2, conv_window=500, 
        patience = 3000, step_cut = 0.1, max_cut = 5, max_iter=int(1e4)):
        """
        Parameters
        ----------
        N: int
            Dimension of signals        
        fwdmodel : object
            instance of a mmv_models.FwdModel class. Object should contain any necessary 
            model-specific parameters as attributes
        sampler : object
            instance of a spore.Sampler class that has a .sample method returning S samples
            of signals X from a probability distribution (N, S, :)            
        batch_size: int
            Number of columns of Y to randomly draw and evaluate for each iteration
        step_size: float 
            initial learning rate for stochastic gradient ascent
        min_lambda: float
            Lower bound on individual entries of lambda. \epsilon in [1]
        pyx_min: float (default 0, i.e. no effect)            
            A batch element y_b is only included in analysis if max(p(y_b|x_s))
            among sampled x's (x_s) is greater than this value. Prevents steps 
            in the direction of junk measurements (e.g. a corrupted siganl) OR 
            if samples are not good for the y_b   
            [1] used 0 for all experiments
        grad_scale: float
            Maximum l2-norm of gradient step that can be taken. Any step larger
            is rescaled to have this l2-norm 
        conv_rel: float (0,1)
            Fractional change in the average of lambda estimate in two conv_windows, 
            below which iteration stops
        conv_window: int
            Number of iterations over which to evaluate moving averages. Nonoverlapping windows
            are compared. E.g. if conv_window = 500, then 999-500 iterations ago is averaged
            and compared to 499-current average.
        patience: int
            Number of iterations to wait for improvement in log likelihood before
            cutting step size
        step_cut: float (0, 1)
            Fraction to cut step size by if patience exceeded
        max_cut: int
            Maximum number of times step size can be cut by step_cut before
            quitting
        max_iter: int
            Maximum iteration budget. SPoRe terminates regardless of convergence status        
        """                
        self.N = N
        if isinstance(fwdmodel, FwdModelGroup):
            self.fwdmodel_group = fwdmodel
        else:
            self.fwdmodel_group = FwdModelGroup([fwdmodel])
        self.sampler = sampler
        self.batch_size = batch_size
        self.step_size = step_size
        self.min_lambda = min_lambda
        self.pyx_min = pyx_min 
        self.grad_scale = grad_scale 
        self.conv_rel = conv_rel
        self.conv_window = conv_window
        self.patience = patience
        self.step_cut = step_cut
        self.max_cut = max_cut
        self.max_iter = max_iter
        
    def recover(self, Y, S, lam0=None, randinit_offset=1e-1, seed=None, verbose=True):
        """Recover poisson rate parameters given 
        
        Parameters
        ----------
        Y : array_like
            Observations.
            Shape ``(M, D)``.
        S : int
            Number of samples to draw for each Y.
        lam0: array_like
            Initial value for estimated lambda. If None, lam0 = randinit_offset
            Shape: ``(N,)
        randinit_offset: float
            Random initializations (if lam0 not provided) are drawn. 
            Offset sets a minimum value for any particular entry of lambda0 
        seed: int or None
            Initial seed for before iterations begin
        verbose: boolean
            If True, prints some information every <self.conv_window> iterations
                 
        Returns
        -------
        lam_S : numpy array
            Recovered estimate of lambda                
            Shape ``(N,)``
        includeCheck: numpy array
            Indices of observations that never influenced a gradient step. These
            observations can be considered 'unexplained' by the recovered lambda. 
            Can be indicative of a corrupted measurement. 
            Not used in [1]
        lamHistory: numpy array
            History of lambda estimates at each iteration            
            Shape ``(N, iters)`` (for iters evaluated until convergence)
        llHistory: numpy array
            History of median log-likelihood estimates at each iteration
            Shape ``(iters,)``
        """

        if isinstance(self.fwdmodel_group, SPoReFwdModelGroup):
            fwdmodel = self.fwdmodel_group
        else:
            _, D = Y.shape
            group_indices = None
            fwdmodel = SPoReFwdModelGroup(self.fwdmodel_group, group_indices)

        M, D = np.shape(Y)
        np.random.seed(seed)
        lamHistory = np.zeros((self.N, self.max_iter))
        llHistory = np.zeros((self.max_iter))        
        
        if lam0 is None:
            lam0 = np.ones(self.N)*randinit_offset
        lamHat = lam0

        # Remaining false elements at convergence => unexplained measurements. Not used in [1]
        includeCheck = np.zeros(D) >  np.ones(D) 
                
        refIter = 0
        bestIter = 0
        stepTemp = self.step_size 
        numCut = 0
        t0 = time.time()

        stepIter = []
        
        # Batch gradient ascent 
        for i in range(self.max_iter):                         
            # Get batch elements and sample for each 
            batchInds = np.random.choice(D, self.batch_size)
            Y_batch = Y[:,batchInds]
            self.sampler._lam = lamHat 
            X_sample = self.sampler.sample(Y_batch, S)        

            pyx = fwdmodel.py_x_batch(Y_batch[:, None, :], X_sample, batchInds) # (S, B) array            

            # Don't eval batch elements whose p(y|x) is too low for all samples. In [1] (self.pyx_min=0)
            batchInclude = np.max(pyx, axis=0) > self.pyx_min  
            includeCheck[batchInds[batchInclude]] = True
            pyx = pyx[:, batchInclude]                
            if np.shape(X_sample)[2] > 1: 
                X_sample = X_sample[:,:,batchInclude]
                
            pqRatio = self.sampler.pq_ratio(X_sample)
            probsAgg = pyx * pqRatio # (S, B) array, aggregate value of pdf computations

            # Evaluate loss and gradient
            llHistory[i] = self.log_likelihood(probsAgg)
            grad = self.gradient(X_sample, lamHat, probsAgg)
            step = stepTemp * grad 

            # Necessary to make more robust against numerical issue described in [1]
            if not np.all(grad==np.zeros(self.N)): # at least some sampled X informs a gradient step
                stepIter.append(i) # track when steps are taken
				
                if np.any( (lamHat+step) >self.min_lambda): #if at least one index is stepped meaningfully 
                    # Rescale according to the indices still in question                                                              
                    normCheck = np.linalg.norm(step[ (lamHat+step) >self.min_lambda]) 
                    if normCheck > self.grad_scale :
                        step = (self.grad_scale / normCheck) * step
                else: # step is likely too big, period. 
                    if np.linalg.norm(step) > self.grad_scale : # Rescale based on whole step vector
                        step = (self.grad_scale / np.linalg.norm(step)) * step                                                        
            #if steps have been taken at least 1/2 the time, recent conv_window worth of iterations likely to have been taken
                # hypothesize that steps may not be taken occasionally at first as lamHat is a bad estimate, but will be taken with increasing regularity                
            enoughSteps = np.sum(np.array(stepIter) > (i - self.conv_window*2)) > self.conv_window 
            
            lamHat += step            
            lamHat[lamHat < self.min_lambda] = self.min_lambda 
            lamHistory[:, i] = lamHat
            
            # Check convergence
            if (i+1) >= (self.conv_window*2):
                lam1 = np.mean(lamHistory[:, (i-2*self.conv_window+1):(i-self.conv_window+1)], axis=1) # e.g [:, 0:500] if conv_window is 500
                lam2 = np.mean(lamHistory[:, (i-self.conv_window+1):(i+1)], axis=1) # e.g. [:, 500:] if i is 999, conv_window is 500
                pctChange = np.linalg.norm(lam2 - lam1, ord=1) / np.linalg.norm(lam1, ord=1)                    
                if pctChange < self.conv_rel and enoughSteps:                     
                    break                                            
                
            # Cut learning rate (if necessary) 
            if llHistory[i] >= llHistory[bestIter] or np.isnan(llHistory[bestIter]): 
                bestIter = i
                refIter = i 
            if i - refIter >= self.patience and enoughSteps:
                stepTemp = self.step_cut * stepTemp
                refIter = i 
                numCut += 1
                if verbose is True: 
                    print('Step size cut ' + str(numCut) + ' times')
            if numCut >= self.max_cut:
                break
                
            # Report: 
            if verbose is True and (i+1)>=(self.conv_window*2) and (i+1) % self.conv_window == 0: 
                print('Iteration #: ' + str(i+1) + '; l1-norm change: ' + str(pctChange) + \
                      '; recovery time: ' + str(round(time.time()-t0, 2)) + ' seconds')                
                
        # average over last conv_window iterations' values             
        lamHat = np.mean(lamHistory[:, (i-self.conv_window+1):(i+1)], axis=1) 
        return lamHat, includeCheck, lamHistory, llHistory
    
    def log_likelihood(self, p_agg): 
        r"""Compute log-likelihood and return the ~average (median/B). 
        Median used because of high variability of individual batch draws. 
        Outlier resistance important if using log-likelihood to inform convergence
        
        Parameters
        ----------
        p_agg: array_like
            element-wise product of p(y|x) (an (S,B,) array) and 
            pqRatio (an (S,B) array or an (S,) array if sample_same=True)
            Explicitly: p_agg for any element is p(y_b|x_s) * p(x_s|\lamHat) / Q(x_s)
            where Q is the sampling function
            Shape: (S, B,)
        Returns
        -------
        ll: average log likelihood of p(y_b|\lambda)        
        """
        S, B = np.shape(p_agg)
        likelihood = (1/S) * np.sum(p_agg, axis=0) # of all batch elements        
        ll = np.median(np.log(likelihood)) / B 
        return ll
    
    def gradient(self, X_s, lamHat, p_agg): 
        """
        Compute MC gradients based on pre-computed measurement/sampling likelihoods
        p(y|x), Q(x_s) (p_agg) and Poisson likelihoods (samples X_s, current estimate lamHat)
        
        Parameters
        ----------
        X_s : array_like
            Sampled X's
            Shape (N, S, B) or (N, S, 1)
        lamHat : array_like 
            current estimate of lambda. Shape (N,)
        p_agg : see log_likelihood()
            
        Returns
        -------
        grad: array_like 
            batch gradient
            Shape: (N,)
        """
        _, _, sameSamples = np.shape(X_s) #same samples over each iteration
        S, B = np.shape(p_agg)
        grad = np.zeros((self.N,))
        
        #Note - it's ok if grad = 0 if all sumChecks fail - equates to waiting 
        #until next iter
        sums = np.sum(p_agg, axis=0)
        sumCheck = sums !=0        
        
        if np.size(sumCheck) != 0: #else just return zero vector
            
            if sameSamples == 1: 
                xOverL = X_s[:,:,0] / lamHat[:, None] #(N, S)             
                grad = np.sum((xOverL @ p_agg[:, sumCheck]) / sums[sumCheck] - 1 , axis=1)
            
            else: 
                xOverL = X_s / lamHat[:, None, None]  #(N, S, B)  
                numer = np.einsum('ij...,j...->i...', xOverL[:,:,sumCheck], p_agg[:,sumCheck])
                grad = np.sum((numer / sums) - 1, axis=1)
                
            grad = grad/B

        return grad
    

class Sampler(ABC):

    @abstractmethod
    def sample(self, Y, S, seed=None):
        """Generate samples of X for each column of Y

        Parameters
        ----------
        Y : array_like
            Observations to sample according to. This array must have
            shape ``(M, B)``.
        S : int
            Number of samples to draw for each Y.
        seed: Random seed for drawing
        Returns
        -------
        X : (N, S, B) or (N, S, 1) ndarray
            S Samples of X for each of B columns of Y. Last dimension is 1 if 
            same samples apply to all batch elements        
        """
        pass
    
    @abstractmethod
    def pq_ratio(self, X): 
        """        
        Get the ratio of probability densities of input X 
            P(X|self._lam)/Q(X) element-wise
            Where P(X|self._lam) is the Poisson probability of each entry in X
            Q(X) is the sampler's probability of drawing that X        
        Parameters
        ----------
        X : array_like
            N-dimensional Vectors within range of Sampler.sample(), stacked in columns of array
            Shape: ``(N, S, B)``  or ``(N, S, 1)``                  
        Returns        
        -------
        ratio : array_like
            Probability densities Q(x) for all X
            Shape: ``(S, B)``
        """        
        pass

class PoissonSampler(Sampler):
    def __init__(self, lam, sample_same=True, seed=None):
        """
        As used in [1]: Q(x) = P(x|lamHat)
        Parameters
        ----------
        lam : array_like (float) 
            Poisson rates from which to draw
            Shape: ``(N,)``
        sample_same : bool
            Whether to use the same X samples for each column of Y.
        
        """
        self._lam = lam
        self._sample_same = sample_same
        self._generator = np.random.default_rng(seed)


    def sample(self, Y, S):
        N, = self._lam.shape
        _, B = Y.shape

        if self._sample_same:
            X = self._generator.poisson(self._lam[:, None, None], (N, S, 1))
        else:
            X = self._generator.poisson(self._lam[:, None, None], (N, S, B))
        
        return X
    
    def pq_ratio(self, X):         
        _, S, B = np.shape(X)            
        #With Poisson sampler - always sampling according to the current lambda value in the sampler 
        ratio = np.ones((S,B))                      
        return ratio