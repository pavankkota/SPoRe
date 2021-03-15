"""
Models for Mutliple Measurement Vector with Poisson constraints (MMVP) 
Linear models most relevant to compressed sensing applications, but other models can be defined

Each model should define the mapping from signal-> measurement (x2y) and the probability
model for measurements (p(y|x) = py_x)
    
Model objects should contain attributes for any internal model parameters, e.g. 
linear models containing a sensing matrix Phi (self.phi) that controls x2y

Authors: Pavan Kota, Daniel LeJeune

Reference: 
[1] P. K. Kota, D. LeJeune, R. A. Drezek, and R. G. Baraniuk, "Extreme Compressed 
Sensing of Poisson Rates from Multiple Measurements," Mar. 2021.

arXiv ID:
"""
from abc import ABC, abstractmethod, abstractproperty

import numpy as np 
from scipy.stats import poisson
from itertools import product
import pdb


class FwdModel(ABC): 
    """
    General class for a forward model, i.e., signal-to-measurement probabilistic 
    mapping x->y with p(y|x)
    Any instantiation of the forward model should self-contain any 
    model-specific parameters, e.g. self.Phi for a linear model in compressed 
    sensing
    """
    @abstractmethod
    def x2y(self, X): 
        """Generate measurements Y for each column of X 
        Instantiates Y given the model and X.    
        
        Parameters
        ----------
        X : array_like
            Signals to map to measurements. 
            Shape ``(N, ...)``.

        Returns
        -------
        Y : array_like
            Measurements Y generated from signals X. 
            Shape ``(M, ...)``
        """        
        pass
    
    @abstractmethod
    def py_x(self, Y, X): 
        """Generate likelihoods of each of ``Y``'s in a batch given the ``X``'s

        Parameters
        ----------
        Y : array_like
            Measurement vectors in batch.
            Shape ``(M, ...)`` starting with DL's model implementations
        X : array_like
            Sampled N-dimensional nonnegative integer vectors. 
            Shape ``(N, ...)``.

        Returns
        -------
        pyx : (...) array_like
            Likelihood of measurements in batch for each sample
        """
        pass

    @abstractproperty
    def output_dim(self):
        """Return the shape ``(M,)`` of a single measurement ``Y`` 

        Returns
        -------
        shape : tuple
            shape ``(M,)`` of a single measurement ``Y``.
        """
        pass


class FwdModelGroup(FwdModel): 
    """Apply multiple (G) forward models to the signals based on their sensor group 
    assignments. G = number of sensor groups. Generalizes to G=1 as well.
    """
    
    def __init__(self, fwdmodels): 
        """Parameters
        ----------
        fwdmodels : list
            list of G objects implementing FwdModel  
        """ 

        if not all(fm.output_dim == fwdmodels[0].output_dim for fm in fwdmodels):
            raise ValueError('All FwdModel objects must have same output dimension.')
            # Room to update/generalize here if desired - different M for each group
            
        self.fms = fwdmodels
        self.G  = len(fwdmodels)
        
    def x2y(self, X, group_indices=None):
        """Generate ``Y`` from ``X`` under the noise model

        Parameters
        ----------
        X : array_like
            Signals to map to measurements. 
            Shape ``(N, ...)``.
        group_indices : array_like
            Indices of which forward model to apply for each signal in ``X``. 
            Broadcastable. If ``None``, use only the first forward model.
            Shape ``(...)``. 
        
        Returns
        -------
        Y : array_like
            Measurements Y generated from signals X.
            Shape ``(M, ...)``
        """

        # use first forward model if no groups specified
        if group_indices is None:
            return self.fms[0].x2y(X)

        else:
        
            N = X.shape[0]

            # create broadcast output shape
            out_shape = self.output_dim + tuple(np.maximum(X.shape[1:], group_indices.shape))
            Y = np.zeros(out_shape)

            for g in range(self.G):
                g_mask = group_indices == g
                x_mask = np.broadcast_to(g_mask[None, ...], X.shape)
                y_mask = np.broadcast_to(g_mask[None, ...], Y.shape)
                Y[y_mask] = self.fms[g].x2y(X[x_mask].reshape(N, -1)).ravel()

            return Y
        
    def py_x(self, Y, X, group_indices=None):
        """Generate likelihoods of each of ``Y``'s in a batch given the ``X``'s

        Parameters
        ----------
        Y : array_like
            Measurement vectors in batch.
            Shape: ``(M, 1, B)`` starting with DL's model implementations
        X : array_like
            Sampled N-dimensional nonnegative integer vectors.
            Shape: ``(N, S, B)``.
        group_indices : array_like
            Indices of which forward model to apply for each signal in ``X``. 
            Broadcastable. If ``None``, use only the first forward model.
            Shape: ``(B,)``. 
        
        Returns
        -------
        pyx : array_like
            Likelihood of measurements in batch for each of sample.
            Shape: ``(S, B)``
        """

        if group_indices is None:
            return self.fms[0].py_x(Y, X)

        else:

            M, N = Y.shape[0], X.shape[0]

            # create broadcast output shape
            out_shape = tuple(np.maximum(Y.shape[1:], np.maximum(X.shape[1:], group_indices.shape)))                   
            pyx = np.zeros(out_shape)

            for g in range(self.G):
                g_mask = group_indices == g

                # handle broadcasting
                if X.shape[-1] == 1:
                    X_masked = X
                else:
                    X_masked = X[:, :, g_mask]

                if Y.shape[-1] == 1:
                    Y_masked = Y
                else:
                    Y_masked = Y[:, :, g_mask]

                pyx[:, g_mask] = self.fms[g].py_x(Y_masked, X_masked)

            return pyx
    
    @property
    def output_dim(self):
        return self.fms[0].output_dim


class SPoReFwdModelGroup(FwdModelGroup):
    """ Class for recovery with SPoRe. Handles sensor groups with batch
    gradients. Can pass in appropriate group assignments from data (simulated or real)
    """
    def __init__(self, fwdmodel_group, group_indices):

        self.fwdmodel_group = fwdmodel_group
        self.group_indices = group_indices

    def x2y(self, X):
        return self.fwdmodel_group.x2y(X, self.group_indices)

    def py_x(self, Y, X):
        return self.fwdmodel_group.py_x(Y, X, self.group_indices)

    def py_x_batch(self, Y, X, batch_indices):

        if self.group_indices is None:
            group_indices = None
        else:
            group_indices = self.group_indices[batch_indices]
            
        return self.fwdmodel_group.py_x(Y, X, group_indices)
    
    @property
    def output_dim(self):
        return self.fwdmodel_group.output_dim


class AutoSPoReFwdModelGroup(SPoReFwdModelGroup):
    """ Automatically assigns groups to measurement indices roughly equally. 
    Particularly useful in simulations 
    """
    def __init__(self, fwdmodels, D):

        group_indices = np.zeros(D)
        for i, partition in enumerate(np.array_split(np.arange(D), len(fwdmodels))):
            group_indices[partition] = i
        super().__init__(FwdModelGroup(fwdmodels), group_indices)


class BaseGaussianNoise(FwdModel):
    """ 
    Fast computations for Gaussian likelihoods
    """
    @abstractmethod
    def mean(self, X):
        pass
    
    @abstractmethod
    def covariance_inv_det(self, X):
        pass

    @abstractmethod
    def covariance_sqrt(self, X):
        pass

    def x2y(self, X):

        mu = self.mean(X)
        Sigma_sqrt = self.covariance_sqrt(X)

        return _gaussian_sample(mu, Sigma_sqrt)

    def py_x(self, Y, X):
        
        mu = self.mean(X)
        Sigma_inv, det = self.covariance_inv_det(X)

        return _gaussian_pdf(Y, mu, Sigma_inv, det)


class LinearWithFixedGaussianNoise(BaseGaussianNoise):

    def __init__(self, Phi, Sigma, full_covariance=False):

        self.Phi = Phi
        self.Sigma = Sigma
        self.full_covariance = full_covariance

        self.Sigma_inv, self.Sigma_det = _compute_covariance_inv_det(self.Sigma, self.full_covariance)
        self.Sigma_sqrt = _compute_covariance_sqrt(self.Sigma, self.full_covariance)
        
    def mean(self, X):
        return np.einsum('ij,j...->i...', self.Phi, X)
    
    def covariance_inv_det(self, X):
        return _add_dims(self.Sigma_inv, X.ndim - 1), _add_dims(self.Sigma_det, X.ndim - 1)
    
    def covariance_sqrt(self, X):
        return _add_dims(self.Sigma_sqrt, X.ndim - 1)
    
    @property
    def output_dim(self):
        return self.Phi.shape[:1]


class LinearWithScalingVarianceGaussianNoise(LinearWithFixedGaussianNoise):

    def __init__(self, Phi, Sigma, scale_factor=1, full_covariance=False):

        self.Phi = Phi
        self.Sigma = Sigma
        self.scale_factor = scale_factor
        self.full_covariance = full_covariance
    
    def covariance_inv_det(self, X):
        Sigma = self.get_scaled_covariance(X)
        return _compute_covariance_inv_det(Sigma, self.full_covariance)
    
    def sqrt_covariance(self, X):
        Sigma = self.get_scaled_covariance(X)
        return _compute_covariance_sqrt(Sigma, self.full_covariance)
    
    def get_scaled_covariance(self, X):

        Sigma = _add_dims(self.Sigma, X.ndim - 1)

        if self.full_covariance:
            return Sigma + self.scale_factor * _diag(self.mean(X))
        else:
            return Sigma + self.scale_factor * self.mean(X)


class PhiGenerator(ABC):
    """Methods for generating sensing matrices for MMV compressed sensing.
    
    """    
    def __init__(self, Mper, N, G=1): 
        """
        Parameters
        ----------
        Mper: int
            Number of sensing channels per sensing matrix
        N: int 
            Number of library elements (atoms) 
        G: int 
            Number of times to split observations between sensing matrices. 
            Each of G sensing matrices will be applied to roughly D/G observations            
        """
        self.Mper = Mper
        self.N = N
        self.G = G    
        
    @abstractmethod 
    def phi_gen(self): 
        """Generate phi         
        Returns
        -------
        phi : (M, N, G) array. 
        """
        pass
    
    def phi_split(self, phiStack):         
        # Split them into G groups 
        phi = np.zeros((self.Mper, self.N, self.G))
        indsSensors = np.array_split(np.arange(self.Mper*self.G), self.G)    
        for g in range(self.G):                 
            #subPhi = np.reshape(phiStack[indsSensors[g],:], (self.Mper, self.N, 1))                               
            phi[:,:,g] = phiStack[indsSensors[g],:]           
        return phi 
    
    def gen_trials(self, numTrials, seed=None): 
        """    
        Parameters
        ----------
        numTrials : int
            Number of trials to generate sensing matrices for
        seed : int, optional
            Random seed initial state. The default is None.
            
        Returns
        -------
        allPhi : array_like
            All sets of sensing matrices for trials. 
            Shape: ``(M, N, G, numTrials)``

        """
        np.random.seed(seed)        
        allPhi = np.zeros((self.Mper, self.N, self.G, numTrials))
 
        for i in range(numTrials):             
            allPhi[:,:,:,i] = self.phi_gen() # group indices won't change
            
        return allPhi


class PhiUniform(PhiGenerator):             
    def phi_gen(self):
        phiStack = np.random.uniform(size=(self.Mper*self.G, self.N))             
        return self.phi_split(phiStack)

class PhiUniformCustomBounds(PhiGenerator):                 
    def __init__(self, Mper, N, G=1, bounds=(0,1,) ): 
        """        
        New Parameters
        ----------
        bounds: tuple with two floats (default (0,1,)) for the lower and upper bound to which
        to rescale a random uniform draw
        """
        self.Mper = Mper
        self.N = N
        self.G = G    
        self.bounds = bounds
        
    def phi_gen(self):
        phiStack = np.random.uniform(size=(self.Mper*self.G, self.N))            
        phiStack = phiStack * (self.bounds[1]-self.bounds[0]) + self.bounds[0]
        return self.phi_split(phiStack)
    
class PhiGaussianPlusOnes(PhiGenerator):   
    """Gaussian matrices with normalized columns concatenated with a single vector of ones
    Row vector of ones in at least one group guarantees identifiability for maximum likelihood estimation (SPoRe)
    """          
    def __init__(self, Mper, N, G=1, norm_cols=False):
        """        
        New Parameter
        ----------
        norm_cols : boolean, optional
            If True, normalize the columns for the Gaussian rows (i.e. ignoring the row of ones). 
            The default is False.
        """
        self.Mper = Mper
        self.N = N
        self.G = G    
        self.norm_cols = norm_cols
        
    def phi_gen(self):
        phiStack = np.random.normal(size=((self.Mper*self.G)-1, self.N)) 
        #phiStack = phiStack / np.linalg.norm(phiStack, axis=0)                
        phiStack = np.concatenate((np.ones((1,self.N)), phiStack), axis=0)
        phis = self.phi_split(phiStack)        
        
        if self.norm_cols is True: 
            for g in range(self.G): 
                phis[:,:,g] = phis[:,:,g] / np.linalg.norm(phis[:,:,g], axis=0)
            
        return phis
        
class PhiGaussianPlusUnif(PhiGaussianPlusOnes):   
    """Gaussian matrices with normalized columns concatenated with a single vector drawn from a uniform random distribution (0,1)
    Row vector of positive values in at least one group guarantees identifiability for maximum likelihood estimation (SPoRe)
    """                  
    def phi_gen(self):
        phiStack = np.random.normal(size=((self.Mper*self.G)-1, self.N)) 
        phiStack = np.concatenate((np.random.uniform(size=(1,self.N)), phiStack), axis=0)
        phis = self.phi_split(phiStack)        
        
        if self.norm_cols is True: 
            for g in range(self.G): 
                phis[:,:,g] = phis[:,:,g] / np.linalg.norm(phis[:,:,g], axis=0)
                
        return phis

def py_lam(fm, yrange, xMax, N, lamInput):
    
    # Generate array of all x's to test
    x_i = [np.arange(xMax+1)]*N
    xTest = np.array(list(product(*x_i))).T #     
    xTest = xTest[:,:,None] # (N, S, B) or (N, S, 1)
    
    #pyx = fm.py_x(yrange, xTest) #(S, B)
    
    pyx = fm.fwdmodel_group.fms[0].py_x(yrange, xTest)
    
    pxlam = np.product(poisson.pmf(xTest[:,:,0], lamInput[:,None]), axis=0) #(S, ...)
    pylam = np.sum(pyx * pxlam[:,None], axis=0)
    return pylam



def _invert_det_batch(A):

    A_rot = np.einsum('ij...->...ij', A)

    A_inv_rot = np.linalg.inv(A_rot)
    A_inv = np.einsum('...ij->ij...', A_inv_rot)

    A_det = np.linalg.det(A_rot)

    return A_inv, A_det

def _sqrtm_batch(A):

    A_rot = np.einsum('ij...->...ij', A)
    w, v = np.linalg.eigh(A_rot)

    return np.einsum('...ij,...kj->ik...', v * np.sqrt(w)[..., None, :], v)
    
def _compute_covariance_inv_det(Sigma, full_covariance):

    if full_covariance:
        inv, det = _invert_det_batch(Sigma)
    else:
        inv = 1 / Sigma
        det = np.prod(Sigma, axis=0)
    
    return inv, det

def _compute_covariance_sqrt(Sigma, full_covariance):

    if full_covariance:
        return _sqrtm_batch(Sigma)
    else:
        return np.sqrt(Sigma)

def _diag(A):

    B = np.zeros(A.shape[:1] + A.shape)
    diag_inds = (np.arange(A.shape[0]),) * 2 + (slice(None),) * (A.ndim - 1)
    B[diag_inds] = A
    return B

def _gaussian_sample(mu, Sigma_sqrt):

    # if full covariance
    if Sigma_sqrt.ndim == mu.ndim + 1:
        out_shape = np.maximum(mu.shape, Sigma_sqrt.shape[1:])
        Z = np.random.randn(*out_shape)
        X = mu + np.einsum('ij...,j...->i...', Sigma_sqrt, Z)
    # if diagonal covariance
    elif Sigma_sqrt.ndim == mu.ndim:
        out_shape = np.maximum(mu.shape, Sigma_sqrt.shape)
        Z = np.random.randn(*out_shape)
        X = mu + Sigma_sqrt * Z
    else:
        raise ValueError('invalid number of dimensions for covariance')

    return X 


def _gaussian_pdf(X, mu, Sigma_inv, det):
    """Gaussian pdf evaluated at ``X`` for mean ``mu`` and possibly 
    ``X``- or ``mu``-dependent covariance ``Sigma``.

    Parameters
    ----------
    X : array_like
        Gaussian observations. Shape ``(M, ...)``.
    mu : array_like
        Mean parameters for each observation. Shape ``(M, ...)``.
    Sigma_inv : array_like
        Inverse covariance parameters for each collection of ``M`` observations. 
        Shape ``(M, M, ...)`` or ``(M, ...)`` if diagonal. For diagonal
        covariance, Sigma contains only the diagonal elements and not the
        surrounding zeros.
    det : array_like
        Determinants of covariance parameters Sigma for each collection of ``M``
        observations.
        Shape ``(...)``.
    
    Returns
    -------
    pdfs : array_like
        Shape ``(...)``.
    """

    M = X.shape[0]

    Diffs = X - mu

    # if full covariance
    if Sigma_inv.ndim == X.ndim + 1:
        Mahalanobis2 = np.einsum('i...,ij...,j...->...', Diffs, Sigma_inv, Diffs)
    # if diagonal covariance
    elif Sigma_inv.ndim == X.ndim:
        Mahalanobis2 = np.einsum('i...,i...->...', Diffs, Sigma_inv * Diffs)
    else:
        raise ValueError('invalid number of dimensions for covariance')

    return np.exp(-Mahalanobis2/2) / np.sqrt((2 * np.pi) ** M * det)


def _add_dims(A, ndim):
    return A.reshape(A.shape + ((1,) * ndim))
