# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:44:28 2020

@author: pkk24
"""

import os
import pickle 
import numpy as np 

import pdb
from itertools import repeat
from multiprocessing import freeze_support, Pool
import time


from spore import spore
from baselines.baselines_utils import process_AutoSPoReFMG_linear as proc
from baselines import algos, baselines_utils
                
def parSPoRe(sporeObject, t, Y, S, seed=None): 
    lamS, _, _, _ = sporeObject.recover(Y, S, seed=seed)     
    print('Trial # ' + str(t+1) + ' complete')
    return lamS

def parSumPoiss(algo, t, Y, fm, xT): 
    Phis, gi, pyxf = proc(fm)            
    _, D = Y.shape
    recInfo = algo.rec(Y, Phis, pyx_func=pyxf, group_indices=gi, xTrue=xT)        
    print('Sum Poisson Trial # ' + str(t+1) + ' complete')     
    return recInfo

def parPoissAlt(algo, t, Y, lam0, fm, xT, label='unspecified'): 
    Phis, gi, pyxf = proc(fm)            
    _, D = Y.shape
    recInfo = algo.rec(Y, Phis, lam0, group_indices=gi, xTrue=xT)            
    print('Poisson Alt Trial # ' + str(t+1) + ' complete; Initializer: ' + str(label))     
    return recInfo


if __name__ == '__main__':         
    freeze_support()
    #allVar = np.logspace(-2, 0, 5)
    simsDataFiles = ['20-11-23_rng1_phiUnif_Var0.001_M2_N10_k3_D100_lamTot2_G1_50trials.pkl', \
                     '20-11-23_rng1_phiUnif_Var0.0021544346900318843_M2_N10_k3_D100_lamTot2_G1_50trials.pkl', \
                     '20-11-23_rng1_phiUnif_Var0.004641588833612777_M2_N10_k3_D100_lamTot2_G1_50trials.pkl', \
                     '20-11-23_rng1_phiUnif_Var0.01_M2_N10_k3_D100_lamTot2_G1_50trials.pkl', \
            '20-11-23_rng1_phiUnif_Var0.021544346900318832_M2_N10_k3_D100_lamTot2_G1_50trials.pkl', \
                     '20-11-23_rng1_phiUnif_Var0.046415888336127774_M2_N10_k3_D100_lamTot2_G1_50trials.pkl', \
                         '20-11-23_rng1_phiUnif_Var0.1_M2_N10_k3_D100_lamTot2_G1_50trials.pkl', \
                             '20-11-23_rng1_phiUnif_Var0.21544346900318823_M2_N10_k3_D100_lamTot2_G1_50trials.pkl', \
                                 '20-11-23_rng1_phiUnif_Var0.46415888336127775_M2_N10_k3_D100_lamTot2_G1_50trials.pkl', \
                                     '20-11-23_rng1_phiUnif_Var1.0_M2_N10_k3_D100_lamTot2_G1_50trials.pkl']
        
    saveStem = 'results_Nov26fix_'
    #notes = ''
    notes = '20/11/23 11:23 AM: Integer baseline algorithms - Redoing with new grad clipping and doing 50 trials'    
    mainSeed = 1
    randInitOffset = 0.1    
    for f in range(len(simsDataFiles)): 
            
        with open(simsDataFiles[f], 'rb') as file: 
            allX, allLam, allY, allFwdModels = pickle.load(file)
        N, numTrials = allLam.shape
        M, D, _ = allY.shape
        k = np.sum(allLam[:,0] !=0)
        lamTot = np.sum(allLam[:,0])
        sampler = spore.PoissonSampler(np.zeros(N), sample_same=True, seed=mainSeed)
        sigInv = baselines_utils.get_sigma_inv_linfixgauss(allFwdModels[0].fwdmodel_group) # assume that all in the file have constant noise setting
            # would not work for scaling noise
        
        oracleAlgo = algos.oracle_int_MMV(k, np.max(allX))
        sumPoissAlgo = algos.SumPoissonSMV(sigInv, alpha=1e-2, lambda_total = lamTot, max_iter_bb=100)
        altAlgo = algos.PoissonAlt(sigInv, alpha=1e-2, max_iter_bb=100, max_alt=10) # 3 different initializers
            #- random initializer [knows k]        
            #- unbiased initializer (lamTot/N) [knows lamTot]
            #- initialize with SumPoissonSMV [knows lamTot]
            #- initialize with SPoRe
        
        #randomLam0 = np.random.uniform(size=(N,numTrials))+randInitOffset
        randomLam0s = []
        for t in range(numTrials):
            randomLam0s.append(np.random.uniform(size=(N,)) + randInitOffset)
        unbiasedLam0 = np.ones(N)*(lamTot/N)
        
        baselineAlgos = [oracleAlgo, sumPoissAlgo, altAlgo]
        B = 6
        
        #SPoRe parameters
        cpuCount = 4
        S = 1000
        convRel = 1e-2
        stepSize = 1e-1
        patience=3000
        maxIterSPoRe=int(1e5)
        gradClip=5e-2
        

        sporeObjs = []
        Ys = []
        xTrues = []
                    
        for t in range(numTrials): 
            sporeObjs.append(spore.SPoRe(N, allFwdModels[t], sampler, conv_rel=convRel, \
                                       max_iter=maxIterSPoRe, step_size=stepSize, patience=patience, grad_clip=5e-2))
            Ys.append(allY[:,:,t])        
            xTrues.append(allX[:,:,t])
        
        
        timepoints = []            
        timepoints.append(time.time())    
        p = Pool(cpuCount)                            
        lamSPoRe = p.starmap(parSPoRe, zip(sporeObjs, np.arange(numTrials), Ys, repeat(S), repeat(mainSeed))) 
        timepoints.append(time.time())
        p.close()
        p.join()
        
 
        # Run baselines
        XB = np.zeros((N, D, numTrials, B))
        lamB = np.zeros((N, numTrials, B))
        timeBaselines = np.zeros((numTrials, B))
        exitFlags = np.zeros((numTrials, B))
        lossFlags = np.zeros((numTrials, B))
 
        for t in range(numTrials): 
            Phis, gi, pyxf = proc(allFwdModels[t])            
            XB[:,:,t, 0] = oracleAlgo.rec(allY[:,:,t], Phis)  
        timepoints.append(time.time())    
        
        # Sum poisson baseline    
        p = Pool(cpuCount)                                    
        sumPoissData = p.starmap(parSumPoiss, zip(repeat(sumPoissAlgo), np.arange(numTrials), Ys, allFwdModels, xTrues)) 
        timepoints.append(time.time())
        p.close()
        p.join()
        lamSum = []
        for t in range(numTrials): 
            lamSum.append(np.sum(sumPoissData[t][0], axis=1)/D)
                    
        #Alternating baselines        
        p = Pool(cpuCount)                            
        altRandData = p.starmap(parPoissAlt, zip(repeat(altAlgo), np.arange(numTrials), Ys, randomLam0s, allFwdModels, xTrues, repeat('Random')))         
        timepoints.append(time.time())
        altUnbData = p.starmap(parPoissAlt, zip(repeat(altAlgo), np.arange(numTrials), Ys, repeat(unbiasedLam0), allFwdModels, xTrues, repeat('Unbiased')))                 
        timepoints.append(time.time())
        altSumPoissData = p.starmap(parPoissAlt, zip(repeat(altAlgo), np.arange(numTrials), Ys, lamSum, allFwdModels, xTrues, repeat('SumPoisson')))         
        timepoints.append(time.time())
        altSPoReData = p.starmap(parPoissAlt, zip(repeat(altAlgo), np.arange(numTrials), Ys, lamSPoRe, allFwdModels, xTrues, repeat('SPoRe')))                 
        timepoints.append(time.time())
        p.close()
        p.join()

        for t in range(numTrials): 
            XB[:,:,t, 1], exitFlags[t,1], lossFlags[t,1] = sumPoissData[t]
            XB[:,:,t, 2], exitFlags[t,2], lossFlags[t,2] = altRandData[t]
            XB[:,:,t, 3], exitFlags[t,3], lossFlags[t,3] = altUnbData[t]
            XB[:,:,t, 4], exitFlags[t,4], lossFlags[t,4] = altSumPoissData[t]
            XB[:,:,t, 5], exitFlags[t,5], lossFlags[t,5] = altSPoReData[t]
            for b in range(B):
                lamB[:,t,b] = np.sum(XB[:,:,t,b], axis=1)/D    


        # Store results 
        lamCosSim = np.zeros((numTrials, B+1))
        lamRelL2err = np.zeros((numTrials, B+1))
        for t in range(numTrials): 
            lamCosSim[t, 0] = np.dot(lamSPoRe[t], allLam[:,t]) / (np.linalg.norm(lamSPoRe[t])*np.linalg.norm(allLam[:,t]))
            lamRelL2err[t, 0] = np.linalg.norm(lamSPoRe[t] - allLam[:,t]) / np.linalg.norm(allLam[:,t])
            
            lamCosSim[t, 1:] = (lamB[:,t,:].T @ allLam[:,t]) / (np.linalg.norm(lamB[:,t,:], axis=0) * np.linalg.norm(allLam[:,t]))
            lamRelL2err[t, 1:] = np.linalg.norm(lamB[:,t,:] - allLam[:,t][:,None], axis=0) / np.linalg.norm(allLam[:,t])
            
        algoTimes = np.diff(timepoints)
        saveResultsFile = saveStem + simsDataFiles[f]
        if os.path.exists(saveResultsFile) is False: 
            with open(saveResultsFile, 'wb') as file: 
                pickle.dump([lamSPoRe, lamB, allLam, lamCosSim, lamRelL2err, cpuCount, algoTimes, baselineAlgos, exitFlags, lossFlags, notes], file)
        else: 
            print('output file already exists - won\'t overwrite')
              
    