# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:47:43 2020

@author: pkk24
"""
import numpy as np 
import pdb
from abc import ABC, abstractmethod
from scipy.special import gammaln, polygamma
from copy import copy, deepcopy
import matplotlib.pyplot as plt

from spore.mmv_models import AutoSPoReFwdModelGroup, FwdModelGroup, LinearWithFixedGaussianNoise

def process_AutoSPoReFMG_linear(model): 
    if not isinstance(model, AutoSPoReFwdModelGroup): 
        raise TypeError('Model type must be AutoSPoReFwdModelGroup')
    
    group_indices = model.group_indices.astype('int')
    G = model.fwdmodel_group.G
    M, N = model.fwdmodel_group.fms[0].Phi.shape
    allPhi = np.zeros((M, N, G))
    pyx_funcs = []
    for g in range(np.max(group_indices)+1): 
        allPhi[:,:,g] = model.fwdmodel_group.fms[g].Phi
        pyx_funcs.append(model.fwdmodel_group.fms[g].py_x)
    
    return allPhi, group_indices, pyx_funcs

def get_sigma_inv_linfixgauss(fwdmodelgroup): 
    if not isinstance(fwdmodelgroup, FwdModelGroup): 
        raise TypeError('Input model type must be FwdModelGroup')
    G = len(fwdmodelgroup.fms)
    M = fwdmodelgroup.fms[0].Phi.shape[0]
    Sigma_inv = np.zeros((M,M,G))
    for g in range(G): 
        if not isinstance(fwdmodelgroup.fms[g], LinearWithFixedGaussianNoise): 
            raise TypeError('Input model\'s component models must be LinearWithFixedGaussianNoise')
        Sigma_inv[:,:,g] = np.diag(fwdmodelgroup.fms[g].Sigma_inv)
    return Sigma_inv
    
class Losses(ABC): 
    def __init__(self):
        pass    
    @abstractmethod
    def loss(self, y, Phi, x): 
        pass
    @abstractmethod
    def grad(self, y, Phi, x): 
        pass
    
class GaussianAndSumPoissonLoss(Losses): 
    """ Loss that penalizes gaussian measurement error (least squares if Sigma
    is diagonal, isotropic) and the sum of each N-dimensional column of X to be 
    Poisson constrained
    
    X can be 1 or 2 dimensional 
    """
    def __init__(self, Y, Phis, Sigma_inv, lamTot, group_indices=None, epsilon=None, w_gauss=1, w_poisson=1):
        self.Y = Y 
        self.Phis = Phis
        self.Sigma_inv = Sigma_inv
        self.lamTot = lamTot
        self.eps = epsilon 
        self.wg = w_gauss
        self.wp = w_poisson
        
        _, D = Y.shape
        if group_indices is None:
            self.group_indices = np.zeros(D)
        else: 
            self.group_indices = group_indices
        
        # Pre-compute and store values useful for gradient         
        _, N, _ = np.shape(Phis)
        self.phiSigPhi = np.zeros((N,N,D))
        self.phiSigY = np.zeros((N,D))
        for d in range(D): 
            self.phiSigPhi[:,:,d] = self.Phis[:,:,self.group_indices[d]].T @ self.Sigma_inv[:,:,group_indices[d]] @ self.Phis[:,:,self.group_indices[d]]
            self.phiSigY[:,d] = self.Phis[:,:,self.group_indices[d]].T @ self.Sigma_inv[:,:,group_indices[d]] @ self.Y[:,d]                
        self.logLamTot = np.log(lamTot)
        
    def loss(self, X):  
        N, D = X.shape        
        if D != self.Y.shape[1] or D != self.Sigma_inv.shape[2]: 
            raise ValueError('Dimension mismatch between input x with stored parameters')
                   
        loss = 0
        for d in range(D):         
            err = self.Phis[:,:, self.group_indices[d]] @ X[:,d] - self.Y[:,d]
            loss += self.wg* (1/2 * err.T @ self.Sigma_inv[:,:,d] @ err) # gaussian error component


        sumX = np.sum(X,axis=0)
        loss += self.wp*(np.sum(gammaln(sumX+1) - sumX*self.logLamTot)) # Poisson component
        
        return loss
    
    def grad(self, X): 
        N, D = X.shape
        if D != self.Y.shape[1] or D != self.Sigma_inv.shape[2]: 
            pdb.set_trace()
            raise ValueError('Dimension mismatch between input x with stored parameters')
            
        grad = np.zeros(X.shape)
        
        for d in range(D):             
            grad[:,d] = self.wg * (self.phiSigPhi[:,:,d] @ X[:,d] - self.phiSigY[:,d]); # gaussian error component

        grad += self.wp* (polygamma(0, np.sum(X, axis=0) + 1) - np.log(self.lamTot)) # (D,) broadcasted across (N,D,) grad. Poisson component
        return grad
                
class GaussianAndPoissonLoss(Losses): 
    """ Loss that penalizes gaussian measurement error (least squares if Sigma
    is diagonal, isotropic) and Poisson loss (given a lambda)
    
    X can be 1 or 2 dimensional 
    """
    def __init__(self, Y, Phis, Sigma_inv, lambda_hat, group_indices=None, epsilon=None, w_gauss=1, w_poisson=1):
        self.Y = Y 
        self.Phis = Phis
        self.Sigma_inv = Sigma_inv
        self.lamHat = lambda_hat
        
        self.eps = epsilon 
        self.wg = w_gauss
        self.wp = w_poisson
        
            
        _, D = Y.shape
        if group_indices is None:
            self.group_indices = np.zeros(D)
        else: 
            self.group_indices = group_indices
        
        # Pre-compute and store values useful for gradient         
        _, N, _ = np.shape(Phis)
        self.phiSigPhi = np.zeros((N,N,D))
        self.phiSigY = np.zeros((N,D))
        for d in range(D): 
            self.phiSigPhi[:,:,d] = self.Phis[:,:,self.group_indices[d]].T @ self.Sigma_inv[:,:,group_indices[d]] @ self.Phis[:,:,self.group_indices[d]]
            self.phiSigY[:,d] = self.Phis[:,:,self.group_indices[d]].T @ self.Sigma_inv[:,:,group_indices[d]] @ self.Y[:,d]                
        self.logLamHat = np.log(self.lamHat+self.eps)[:,None]
        
    
    def loss(self, X):  
        N, D = X.shape        
        if D != self.Y.shape[1] or D != self.Sigma_inv.shape[2]: 
            raise ValueError('Dimension mismatch between input x with stored parameters')
        
        #x = copy(X)
        #if self.eps is not None:
            #x[x < self.eps] = self.eps # some minimum value of X (e.g. X must be > 0)
            
        loss = 0
        for d in range(D):         
            err = self.Phis[:,:, self.group_indices[d]] @ X[:,d] - self.Y[:,d]
            loss += self.wg* (1/2 * err.T @ self.Sigma_inv[:,:,d] @ err) # gaussian error component

        loss += self.wp*(np.sum(gammaln(X+1) - X*self.logLamHat))
        #pdb.set_trace()
        return loss
    
    def grad(self, X): 
        N, D = X.shape
        if D != self.Y.shape[1] or D != self.Sigma_inv.shape[2]: 
            pdb.set_trace()
            raise ValueError('Dimension mismatch between input x with stored parameters')
            
        grad = np.zeros(X.shape)
        
        for d in range(D):             
            grad[:,d] = self.wg * (self.phiSigPhi[:,:,d] @ X[:,d] - self.phiSigY[:,d]); # gaussian error component
        grad += self.wp* (polygamma(0, X+1) - np.repeat( self.logLamHat, D, axis=1) )
        #pdb.set_trace()
        return grad
    
    
    

class BranchAndBound(ABC): 
    # Many recommendations followed from: 
    # P Bonami et al, More Branch-and-Bound Experiments in Convex Nonlinear Integer Programming 
    def __init__(self, optimizer, loss_func): 
        """        
        Parameters
        ----------
        optimizer : function
            Function that has all other parameters defined (e.g. a partial function) 
            except for x, lower bounds, and upper bounds
        loss_func : function to optimize
            - should be the loss function passed to optimizer
        """
        self.optimizer = optimizer
        self.loss_func = loss_func
        
    def solve(self, x0, fullLBs, fullUBs, max_iter=1e4):         
        """
        
        Parameters
        ----------

        x0 : numpy array
            initial value for x
        fullLBs: numpy array
            Array of lower bounds for all values of X. Must be same size as x0
        fullUBs: numpy array
            Array of lower bounds for all values of X. Must be same size as x0
        maxIter : int, optional
            Maximum number of branches to try. The default is 1e5.

        Returns
        -------
        xInt : numpy array
            Integer solution within fullLBs and fullUBs
        exitFlag : boolean
            if True, then true optimum wasn't found before another stopping criteria (e.g. maxIter) was hit
            if False, then all competing branches eliminated
        """        
        if fullLBs.shape != x0.shape or fullUBs.shape != x0.shape: 
            raise ValueError('Dimensions of LBs and UBs must match x0')
                
        counter=0
        branchList = np.array([np.array((fullLBs, fullUBs, np.inf))])# bounds and the lower bound of parent branch

        bestX = copy(np.round(x0))        
        solUB = self.loss_func(bestX)
        
        #solUB = np.inf      
        solUBupdate=[]
        exitFlag = False
        
        while counter < max_iter and np.shape(branchList)[0] > 0:  
            branchIndex = self.choose_branching_index(branchList)
            if counter > 0: 
                x0 = xConvex #last convex solution
            # Evaluate chosen branch           
            if np.any(branchList[branchIndex][0] != branchList[branchIndex][1]): 
                xConvex, xCvxLoss = self.optimizer(x0, LBs=branchList[branchIndex][0], UBs=branchList[branchIndex][1]) 
            else: 
                xConvex = branchList[branchIndex][0] # bounds are equal to each other, no need to optimize
                xCvxLoss = self.loss_func(xConvex)

            if xCvxLoss > solUB: # prune branch
                branchList = np.delete(branchList, branchIndex, axis=0)
                #branchList.pop(branchIndex)
            
            else: # branch is worth exploring
                indsZ = (xConvex == np.round(xConvex))
                if np.sum(indsZ) == np.size(xConvex): # branch is done
                    solUB = xCvxLoss
                    solUBupdate.append(counter)
                    bestX = copy(xConvex)
                    branchList = np.delete(branchList, branchIndex, axis=0)
                    #branchList.pop(branchIndex)
                    continue
                else: 
                    # may as well test a rounded solution (cheap way to potentially reduce solUB, prune branches faster)
                    roundLoss = self.loss_func(np.round(xConvex))
                    if roundLoss < solUB:
                        solUB = roundLoss
                        solUBupdate.append(counter)
                        bestX = np.round(copy(xConvex))
                    
                                    
                    # Add branches, remove the original 
                    branchingVariable =  self.choose_branching_variable(branchList, branchIndex, xConvex, xCvxLoss)
                    if branchingVariable != -1:
                        branchList = self.add_branches(branchList, branchIndex, xConvex, xCvxLoss, branchingVariable)                                                   
                    else: #error in probabilities
                        print('WARNING: probabilities error')                        
                        branchList = np.delete(branchList, branchIndex, axis=0)
                        
                        
                                              
            counter+=1

        # TO DO exit flag
        if counter == max_iter:
            exitFlag = True
        #pdb.set_trace()
        return bestX, exitFlag
    
    def choose_branching_index(self, branchList): 
        return np.argmin(branchList[:,2]) # minimum convex loss of parent        

    def add_branches(self, branchList, branchIndex, xConvex, xCvxLoss, varIndex):                      
        # create new branches
        bTemp1 = deepcopy(branchList[branchIndex])
        bTemp2 = deepcopy(branchList[branchIndex])
        bTemp1[0][varIndex[0], varIndex[1]] = np.ceil(xConvex[varIndex[0], varIndex[1]]) # update lower bound on one branch
        bTemp2[1][varIndex[0], varIndex[1]] = np.floor(xConvex[varIndex[0], varIndex[1]]) # update upper bound on one branch        
        bTemp1[2] = xCvxLoss
        bTemp2[2] = xCvxLoss
        
        # add new branches and remove old branch
        branchList = np.concatenate((branchList, bTemp1[None,:]), axis=0)
        branchList = np.concatenate((branchList, bTemp2[None,:]), axis=0)
        branchList = np.delete(branchList, branchIndex, axis=0)
        return branchList
    
    @abstractmethod
    def choose_branching_variable(self): 
        pass    
    

    
class BranchAndBoundMostFractional(BranchAndBound): 
    def choose_branching_variable(self, branchList, branchIndex, xConvex, xCvxLoss): 
        # Choose the branching *variable*         
        roundDist = np.absolute(xConvex - np.round(xConvex))
        probs = roundDist/np.sum(roundDist)
        
        if np.any(np.isnan(probs)):            
            return -1
        else:
            index2branch = np.random.choice(np.arange(np.size(probs)), p = np.reshape(probs, np.size(probs)))
            inds = np.unravel_index(index2branch, probs.shape)        
            return inds
    

    
def grad_descent(loss_func, grad_func, alpha, x0, nu=0, LBs=None, UBs=None, patience=100, alpha_cut = 3e-1, max_iter=int(1e5), conv_rel=1e-3, grad_tol=1e-3, xTrue=None): 
    """ Simple gradient descent with momentum option with some features to work with 
    BranchAndBound if desired
    
    Parameters
    ----------
    loss_func : method 
        Takes x as input and returns a scalar loss value        
    grad_func : method
        Takes x as input and returns gradient vector (same dimensions as x) 
    x0 : array_like 
        initial value for optimization
    alpha : float
        learning rate
    nu : float (0,1)
        momentum parameter. Default = 0
        
    Returns
    -------
    xHatFinal: numpy array
        Estimated minimizer
    loss_func(xHatFinal): scalar
        Loss value for final xHatFinal    

    """
    if LBs is None: 
        LBs = -np.inf * np.ones(x0.shape)
    if UBs is None: 
        UBs = np.inf * np.ones(x0.shape)
        
    if LBs.shape != x0.shape or UBs.shape != x0.shape: 
        raise ValueError('Dimensions of LBs and UBs must match x0')

    x0[x0 < LBs] = LBs[x0 < LBs]
    x0[x0 > UBs] = UBs[x0 > UBs]
    xHat = x0  
    xHatFinal = copy(x0)
    bestLoss = loss_func(xHatFinal)
    
    vdw = np.zeros(xHat.shape)
    counter = 0
    refIter = 0
    cutIter = 0
    numCuts = 0
    numBoost = 0
    boostIter = 0
    
    lossHistory = np.zeros(max_iter)
    
    
    while counter < max_iter:
        grad = grad_func(xHat)
        vdw2 = nu*vdw + (1-nu)*grad
        #if np.linalg.norm(alpha*vdw2) > grad_clip: 
            #xHatNew = xHat - (alpha*vdw2)*(grad_clip/np.linalg.norm(alpha*vdw2))
        #elif np.linalg.norm(alpha*vdw2) < grad_min: 
            #xHatNew = xHat - (alpha*vdw2)*(grad_min/np.linalg.norm(alpha*vdw2))
        #else: 
        xHatTemp = xHat - alpha*vdw2
        xHatNew = copy(xHatTemp)
        xHatNew[xHatTemp < LBs] = LBs[xHatTemp < LBs]
        xHatNew[xHatTemp > UBs] = UBs[xHatTemp > UBs]
        loss2 = loss_func(xHatNew)
        
        
        #if loss_func(xHatNew) < loss_func(xHatFinal): 
        if loss2 < bestLoss: 
            bestLoss = loss2
            xHatFinal = copy(xHatNew)
            refIter = counter
            
            # check if gradient on the viable values is tiny
            if np.linalg.norm(alpha*vdw2[(xHatTemp>=LBs) * (xHatTemp<=UBs)]) < grad_tol and (counter-cutIter)>patience*2: 
                break
            
        
        # Cut learning rate if necessary 
        if counter - refIter > patience:                   
            alpha = alpha * alpha_cut
            xHat = copy(xHatFinal) # restore current best
            vdw = np.zeros(xHat.shape)
            cutIter = counter
            refIter = counter
            lossHistory[counter] = loss_func(xHatNew)
            counter+=1
            numCuts+=1
            if numCuts > 10: 
                if np.absolute((loss2-lossHistory[counter-1])/lossHistory[counter-1]) < conv_rel: 
                    break
            continue
            
            
        # boost learning rate if possible
        if refIter - cutIter > patience and refIter - boostIter > patience and (numCuts == 0):
            alpha = alpha / alpha_cut
            vdw2 = np.zeros(xHat.shape)            
            numBoost+=1
            boostIter = counter
                            
        if (counter - cutIter) > patience*2:
            lossOld = np.mean(lossHistory[(counter-2*patience+1):(counter-patience+1)]) 
            lossNew = np.mean(lossHistory[(counter-patience+1):(counter)]) 
            pctChange = np.abs(lossOld-lossNew) / np.abs(lossOld)
            if pctChange < conv_rel: 
                break

        if (counter+1) % 10000 == 0: 
            print('10k iterations in GD')
            
        xHat = xHatNew    
        vdw = vdw2
        lossHistory[counter] = loss2
        counter+=1
        
        
    if np.any(xHatFinal < LBs) or np.any(xHatFinal > UBs): 
        #pdb.set_trace()
        print('WARNING: solution out of bounds')

    if xTrue is not None: 
        if np.all(xTrue >= LBs) and np.all(xTrue <= UBs) and loss_func(xHatFinal)-loss_func(xTrue) > np.abs(0.01*loss_func(xTrue)): 
            print('WARNING: optimizer did not find minimum, loss true: ' + str(loss_func(xTrue)) + '; loss found: ' + str(loss_func(xHatFinal)))
    #print('Num iterations: ' + str(counter))

    return xHatFinal, bestLoss

def grad_descent_backtracking(loss_func, grad_func, alpha, x0, ro=0.1, c1=0.5, LBs=None, UBs=None, max_iter=int(1e5), grad_tol=1e-1, conv_window=100, conv_rel=1e-3, xTrue=None): 
    """ Simple gradient descent with backtracking line search with some features to work with 
    BranchAndBound if desired
    
    Parameters
    ----------
    loss_func : method 
        Takes x as input and returns a scalar loss value        
    grad_func : method
        Takes x as input and returns gradient vector (same dimensions as x) 
    x0 : array_like 
        initial value for optimization
    alpha : float
        learning rate
    nu : float (0,1)
        momentum parameter. Default = 0
        
    Returns
    -------
    xHatFinal: numpy array
        Estimated minimizer
    loss_func(xHatFinal): scalar
        Loss value for final xHatFinal    

    """
    if LBs is None: 
        LBs = -np.inf * np.ones(x0.shape)
    if UBs is None: 
        UBs = np.inf * np.ones(x0.shape)
        
    if LBs.shape != x0.shape or UBs.shape != x0.shape: 
        raise ValueError('Dimensions of LBs and UBs must match x0')
        
    x0[x0 < LBs] = LBs[x0 < LBs]
    x0[x0 > UBs] = UBs[x0 > UBs]
    xHatFinal = x0  
    #xHatFinal = copy(x0)
    bestLoss = loss_func(xHatFinal)
    gradConsider = LBs!=UBs    

    lossHistory = np.zeros(max_iter)
    
    counter=0
    while counter < max_iter:
        grad = grad_func(xHatFinal)

        
        if np.linalg.norm(grad[(gradConsider) * (( (xHatFinal > LBs)*(xHatFinal<UBs) ) + ((xHatFinal==LBs) * (grad<0)) + ((xHatFinal==UBs)*(grad>0))) ]) < grad_tol:
            break
        
        # Ultra basic line search - not quite Armijo-Goldstein
        alphaAdjust = alpha         
        adjustCounter = 0
        while 1: 
            xHatNew = xHatFinal - alphaAdjust*grad            
            xHatNew[xHatNew < LBs] = LBs[xHatNew < LBs]
            xHatNew[xHatNew > UBs] = UBs[xHatNew > UBs]
            newLoss = loss_func(xHatNew)            
            if newLoss > bestLoss:
                alphaAdjust*=ro
                adjustCounter+=1
            elif newLoss <= bestLoss and adjustCounter == 0 and not np.all((xHatNew == LBs) + (xHatNew == UBs)): # try taking a larger step
                # if alpha is pushing things to the bounds, then no larger alpha will make a difference                    
                alphaAdjust/=ro                
            else:
                bestLoss = newLoss
                lossHistory[counter] = bestLoss                
                xHatFinal = xHatNew
                counter+=1
                break
        
        if counter > conv_window*2:
            lossOld = np.mean(lossHistory[(counter-2*conv_window+1):(counter-conv_window+1)]) 
            lossNew = np.mean(lossHistory[(counter-conv_window+1):(counter)]) 
            pctChange = np.abs(lossOld-lossNew) / np.abs(lossOld)
            if pctChange < conv_rel:                
                break
            
        
    if counter == max_iter: 
        print('Warning: Gradient descent backtracking reached max iterations')
        
    if np.any(xHatFinal < LBs) or np.any(xHatFinal > UBs): 
        #pdb.set_trace()
        print('WARNING: solution out of bounds')

    if xTrue is not None: 
        if np.all(xTrue >= LBs) and np.all(xTrue <= UBs) and loss_func(xHatFinal)-loss_func(xTrue) > np.abs(0.01*loss_func(xTrue)): 
            print('WARNING: optimizer did not find minimum, loss true: ' + str(loss_func(xTrue)) + '; loss found: ' + str(loss_func(xHatFinal)))
            #pdb.set_trace()
    #print('Num iterations: ' + str(counter))

    return xHatFinal, bestLoss

def gaussianLoss(): 
    pass


