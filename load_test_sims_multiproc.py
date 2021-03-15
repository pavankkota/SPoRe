# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:44:28 2020

@author: pkk24
"""

import os
import pickle 
import numpy as np 
import pdb
#from matplotlib import pyplot as plt

from functools import partial
from itertools import repeat
from multiprocessing import freeze_support, Pool
import time


from spore import spore
from baselines.baselines_utils import process_AutoSPoReFMG_linear as proc
from baselines import algos, baselines_utils
                
def parSPoRe(sporeObject, t, Y, S, lam0, seed=None): 
    #lamS, _, _, _ = sporeObject.recover(Y, S, seed=seed)     
    lamS, _, _, _ = sporeObject.recover(Y, S, lam0=lam0, seed=seed)     
    
    if lam0 is not None: 
        cossim = np.dot(lamS, lam0)/(np.linalg.norm(lam0) * np.linalg.norm(lamS))
        print('Trial # ' + str(t+1) + ' complete; cosine sim with lam0: ' + str(cossim))
    else: 
        print('Trial # ' + str(t+1) + ' complete')
        
    return lamS

if __name__ == '__main__':         
    freeze_support()
    
    #manual file list
    simsDataFiles = ['21-03-14_rng1_phiUnif_Var1e-06_M1_N20_k3_D100_lamTot2_G1_50trials.pkl']
    
    #2-param sweep    
    # simsFilesChunks = ['20-10-26_phiUnif_Var0.01_M2_N50_k', '_D1000_lamTot', '_G1_50trials.pkl']
    # var1 = [3,4,5,6,7]
    # var2 = [1,2,4,6,8,10]
    # simsDataFiles = []
    # for i in range(len(var1)):
    #     for j in range(len(var2)): 
    #         simsDataFiles.append(simsFilesChunks[0]+str(var1[i])+simsFilesChunks[1]+str(var2[j])+simsFilesChunks[2])
    
    saveStem = 'results_'
    #notes = ''
    notes = '21/03/14: Adding an M=1 point to the key M vs performance plot'
    mainSeed = 1

            
    for f in range(len(simsDataFiles)): 
            
        with open(simsDataFiles[f], 'rb') as file: 
            allX, allLam, allY, allFwdModels = pickle.load(file)
        N, numTrials = allLam.shape
        M, D, _ = allY.shape
        k = np.sum(allLam[:,0] !=0)
        lamTot = np.sum(allLam[:,0])
        sampler = spore.PoissonSampler(np.zeros(N), sample_same=True, seed=mainSeed)
        
        # assume that all in the file have constant noise setting
            # Does not yet work for scaling noise
        sigInv = baselines_utils.get_sigma_inv_linfixgauss(allFwdModels[0].fwdmodel_group) 

        # CS baseline list:             
        baselineAlgos = [algos.DCS_SOMP(k_known=k, pre_condition=True, pre_cond_eps=1e-3), algos.PROMP(0.5, pos_constrained=True, k_known=k, pre_condition=True, pre_cond_eps=1e-3), \
                         algos.NNS_SP(k, pre_condition=True, pre_cond_eps=1e-3), algos.NNS_CoSaMP(k, pre_condition=True, pre_cond_eps=1e-3)]
        
        #if just for placeholder when running multiproc sims, DCS_SOMP is quickest:
        #baselineAlgos = [algos.DCS_SOMP(k_known=k, pre_condition=True, pre_cond_eps=1e-3)]
    
        B = len(baselineAlgos)
        
        
        # Define recovery algorithm(s) parameters as needed
        cpuCount = 2
        S = 1000
        convRel = 1e-2
        stepSize = 1e-1
        patience=3000
        maxIterSPoRe=100000
        gradScale=5e-2
        lamTrueInit = False
        
        sporeObjs = []
        Ys = []
        lam0s = [] 
    
        for t in range(numTrials): 
            #sporeObjs.append(spore.SPoRe(N, allFwdModels[t], sampler, conv_rel=convRel, \
                                       #max_iter=maxIterSPoRe, step_size=stepSize, patience=patience))        
            sporeObjs.append(spore.SPoRe(N, allFwdModels[t], sampler, conv_rel=convRel, \
                                       max_iter=maxIterSPoRe, step_size=stepSize, patience=patience, grad_scale=gradScale))
            
            Ys.append(allY[:,:,t])
            
            
            
            if lamTrueInit: 
                lamTemp = allLam[:,t]
                lamTemp[np.where(lamTemp==0)] = sporeObjs[t].min_lambda
                lam0s.append(allLam[:,t])
            else: 
                lam0s.append(None) # if no special initialization
            
        t0 = time.time()

        p = Pool(cpuCount)                                    
        lamS = p.starmap(parSPoRe, zip(sporeObjs, np.arange(numTrials), Ys, repeat(S), lam0s, repeat(mainSeed) )) 
        p.close()
        p.join()
        timeTakenSPoRe = time.time()-t0
    
        
        # Run the baselines (all fast enough for single process)
        XB = np.zeros((N, D, numTrials, B))
        lamB = np.zeros((N, numTrials, B))
        timeBaselines = np.zeros((numTrials, B))
        for t in range(numTrials): 
            err_tol = 1
            Phis, gi, pyxf = proc(allFwdModels[t])

            
            for b in range(B): 
                t0 = time.time()
                XB[:,:,t,b] = baselineAlgos[b].rec(allY[:,:,t], Phis, pyx_func=pyxf, group_indices=gi)
                lamB[:,t,b] = np.sum(XB[:,:,t,b], axis=1)/D
                timeBaselines[t,b] = time.time() - t0
            print('Baselines: trial ' + str(t+1) + ' complete')
        # Store results 
        lamCosSim = np.zeros((numTrials, B+1))
        lamRelL2err = np.zeros((numTrials, B+1))
        for t in range(numTrials): 
            lamCosSim[t, 0] = np.dot(lamS[t], allLam[:,t]) / (np.linalg.norm(lamS[t])*np.linalg.norm(allLam[:,t]))
            lamRelL2err[t, 0] = np.linalg.norm(lamS[t] - allLam[:,t]) / np.linalg.norm(allLam[:,t])
            
            lamCosSim[t, 1:] = (lamB[:,t,:].T @ allLam[:,t]) / (np.linalg.norm(lamB[:,t,:], axis=0) * np.linalg.norm(allLam[:,t]))
            lamRelL2err[t, 1:] = np.linalg.norm(lamB[:,t,:] - allLam[:,t][:,None], axis=0) / np.linalg.norm(allLam[:,t])
            
        saveResultsFile = saveStem + simsDataFiles[f]
        if os.path.exists(saveResultsFile) is False: 
            with open(saveResultsFile, 'wb') as file: 
                pickle.dump([lamS, lamB, allLam, lamCosSim, lamRelL2err, cpuCount, timeTakenSPoRe, timeBaselines, baselineAlgos, notes], file)
        else: 
            print('output file already exists - won\'t overwrite')
              
    