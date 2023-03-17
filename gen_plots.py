# -*- coding: utf-8 -*-
"""
Generate final plots for 

[1] Kota et al, Expanded Multiplexing on Sensor-Constrained Microfluidic Partitioning Systems
    https://www.biorxiv.org/content/10.1101/2022.12.23.521805v1
    
@author: pkk24
"""

import pickle
from os import listdir
import pdb
import pandas as pd
import numpy as np
from spore import mmvp_exact_grad
from scipy.io import loadmat
from scipy.spatial.distance import cosine
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from sklearn.metrics import roc_curve, auc, confusion_matrix

def dist_check(mleObj, lam, groupWeights, D, ddof): 
    """
    Used in flagging samples with out-of-panel barcodes. Evaluate chi sq between 
    expected (based on recovered solution) and observed distributions of measurements
    """

    G = mleObj.G
    lamRecReduced = mleObj.lam_reduce(lam)
    
    nllY = np.zeros((G,4))    
    for g in range(G): 
        nllY[g,:] = mleObj.nlog_py_lam(lamRecReduced[g,:])    
    
    pY = np.exp(-nllY)
    pY = pY * groupWeights[:,None] #actual expected distribution of Y's given the # of droplets in each group
    nllMean = np.sum(np.multiply(pY, nllY))
    nllVar = np.sum(np.multiply(pY, (nllY - nllMean)**2))

    #Empirical:         
    pYempiricalAdjust = pYempirical.T*groupWeights[:,None]
    nllMeanEmpirical = nllCompare[i,0]
    nllVarEmpirical = np.sum(np.multiply(pYempiricalAdjust, (nllY - nllMeanEmpirical)**2))
    
    chival, pval = chisquare(pYempiricalAdjust * D, pY*D, ddof=ddof, axis=None)
    
    return (nllMean, nllVar), (nllMeanEmpirical, nllVarEmpirical), (chival, pval)


"""
Inputs
"""
mainSeed = 1
file = open('processed_raw_data.pkl', 'rb')
allGroupInds, allY = pickle.load(file)
numSamples = len(allY)
gtMat = loadmat(r'ddpcr_data\\ground_truth\\22-07-13_lamTrue.mat')
lamTrue = gtMat['lamTrue']


# sensing matrix for each group (which HEX/FAM probe hits each bacteria)
# order: A baum, B frag, E cloacae, E faecium, E coli, K pneu, P aeru, Staph. Strep.
# Group order: G1: Probes 1,3; G2: Probes 2,3; G3: Probes 1,4; G4: Probes 5,3
ecloac= 0.875
PSI = [np.array([   [0, 1, ecloac, 1, 0, 1, 0, 0, 0], \
                [1, 0, 1, 0, 1, 1, 1, 0, 1] ]), \
   
   np.array([   [0, 0, 0, 0, 0, 0, 1, 1, 0], \
                [1, 0, 1, 0, 1, 1, 1, 0, 1] ]), \
       
   np.array([   [0, 1, ecloac, 1, 0, 1, 0, 0, 0], \
                [1, 0, 1, 0, 0, 0, 1, 0, 0] ]), \
       
   np.array([   [0, 0, 0, 1, 0, 0, 0, 1, 1], \
                [1, 0, 1, 0, 1, 1, 1, 0, 1] ])]  
PSI = np.moveaxis(np.array(PSI), 0, 2)
PSI = PSI.astype(float)
_,N,G = PSI.shape


# MLE settings
Ds = np.array([1, 0.1,  0.01, 0.001, 0.0001]) # subsampling data. If not, just use np.array([1])
Ds = np.array([1]) # subsampling data. If not, just use np.array([1])
convTol = 1e-8 #note this was 1e-6 in the original bioRxiv post, but it was found that lambda wasn't fully converging
p_err = 0
epsilon = 1e-6
groupWeight=True # proportional weighting to groups in optimization.  if True, scale by droplet count
lamTrue2 = deepcopy(lamTrue)
lamTrue2[lamTrue2==0] = epsilon

#chi sq stuff
flagTest = False # if evaluating the ability to flag out-of-panel barcodes. Increases total runtime
ddof=0


"""
Main
"""
truePvals = []
badPvals = []
domPvals = []
pctOmit = []

allFMG = []
sporeObjs = []

lamRec = np.zeros( (N, numSamples, Ds.size))
nllCompare = np.zeros((numSamples, 2))
lamCosSim = np.zeros((numSamples, Ds.size))
estCopyCount = np.zeros((numSamples, Ds.size))


np.random.seed(mainSeed)
for i in range(numSamples): 
    
    for d in range(Ds.size): 
        mleObj = mmvp_exact_grad.BinaryMLE(PSI, p_err)
    
        # Generate subsample from each group
        subS = []
        for g in range(G): 
            inds = np.where(allGroupInds[i]==g)[0]
            subS += list(np.random.choice(inds, size=round(Ds[d]*inds.size), replace=False))
        subS = np.array(subS)
                            
        yTemp = allY[i][:,subS]
        giTemp = allGroupInds[i][subS]
        
        
        lam0 = np.ones((N,))*0.1
        lamRec[:,i,d] = mleObj.recover(yTemp, giTemp, lam0, groupWeight=groupWeight, convtol=convTol, epsilon=epsilon)
        
        lamCosSim[i,d] = 1-cosine(lamRec[:,i,d], lamTrue[:,i])
        estCopyCount[i,d] = np.sum(lamTrue[:,i]) * subS.size
        
        groupWeights = np.zeros(4)
        for g in range(4):
            groupWeights[g] = np.sum(allGroupInds[i]==g) / np.size(allGroupInds[i])
            
        pYempirical = mleObj.get_p_y(allY[i], allGroupInds[i])        
        nllCompare[i,0] = mleObj.NLL(mleObj.lam_reduce(lamRec[:,i,d]), allY[i], groupWeights, pYempirical)    
        nllCompare[i,1] = mleObj.NLL(mleObj.lam_reduce(lamTrue2[:,i]), allY[i], groupWeights, pYempirical)
        
        
        #Now want E[-log p(y|\lambda_rec)], var[-log p(y|\lambda_rec)] versus the empirical distribution
        lamRecReduced = mleObj.lam_reduce(lamRec[:,i,d])
        
        nllY = np.zeros((G,4))    
        for g in range(G): 
            nllY[g,:] = mleObj.nlog_py_lam(lamRecReduced[g,:])    
            #pY[g,:] = mleObj.py_lam(lamRecReduced[g,:])
        
        pY = np.exp(-nllY)
        pY = pY * groupWeights[:,None] #actual expected distribution of Y's given the # of droplets in each group
        nllMean = np.sum(np.multiply(pY, nllY))
        nllVar = np.sum(np.multiply(pY, (nllY - nllMean)**2))
    
        #Empirical:         
        pYempiricalAdjust = pYempirical.T*groupWeights[:,None]
        nllMeanEmpirical = nllCompare[i,0]
        nllVarEmpirical = np.sum(np.multiply(pYempiricalAdjust, (nllY - nllMeanEmpirical)**2))
        
        chival, pval = chisquare(pYempiricalAdjust * allY[i].shape[1], pY*allY[i].shape[1], ddof=ddof, axis=None)
        truePvals.append(pval)
        

        if flagTest and Ds[d] == 1:   
            supp = np.where(lamTrue[:,i] !=0)[0]        
            for k in range(len(supp)): 
                #lamTemp = np.delete(lamTrue[:,i], supp[k])
                PSItemp = np.delete(PSI, supp[k], axis=1)
                lam0temp = np.delete(lam0, supp[k])
                mleObj = mmvp_exact_grad.BinaryMLE(PSItemp, p_err)
                lamRecTemp = mleObj.recover(allY[i], allGroupInds[i], lam0temp, groupWeight=groupWeight, convtol=convTol, epsilon=epsilon)
                
                expNLL, empNLL, chisqinfo = dist_check(mleObj, lamRecTemp, groupWeights, allY[i].shape[1], ddof=ddof)
                badPvals.append(chisqinfo[1])
                pctOmit.append(lamTrue[supp[k], i] / np.sum(lamTrue[:,i]))
                
                if lamTrue[supp[k],i] == np.max(lamTrue[:,i]):
                    domPvals.append(chisqinfo[1])
    
        print('Iteration ' + str(i+1) + '/' + str(numSamples) + ' complete')
    

fs = 14
cmax = 2.5568
#cmax = 3.5
mew = 0.5
ms = 8


plt.figure(dpi=400)
plt.plot(np.arange(18)+1, -nllCompare[:,0], 'r-')
plt.plot(np.arange(18)+1, -nllCompare[:,1], 'k--')
plt.legend([r'Mean $p(y|\lambda)$ (Recovered)', 'Mean $p(y|\lambda)$ (True)'], fontsize=12)
plt.xlabel('Sample number', fontsize=14)
plt.ylabel('Average Log Likelihood', fontsize=14)
#plt.title('Log Likelihood Comparison of Solutions')
plt.xticks(np.arange(18)+1)


plt.figure(dpi=400)
plt.imshow(lamRec[:,:,0], cmap='hot', vmin=0, vmax=2.5568)
plt.title('Recovered Concentration with SPoRe', fontsize=16)
ax = plt.gca()
ax.set_yticklabels(['A. baum.', 'B. frag.', 'E. cloac.', 'E. faec.', 'E. coli', 'K. pneu', \
               'P. aeru', 'Staph.', 'Strep.'], fontsize=14, style='italic')
ax.set_yticks(np.arange(9))
plt.xlabel('Sample Number', fontsize=14)
plt.xticks(np.arange(18), labels=np.arange(18)+1)
plt.colorbar(shrink=0.7)
plt.show()

plt.figure(dpi=400)
plt.imshow(lamTrue, cmap='hot', vmin=0, vmax=2.5568)
ax = plt.gca()
ax.set_yticklabels(['A. baum.', 'B. frag.', 'E. cloac.', 'E. faec.', 'E. coli', 'K. pneu', \
               'P. aeru', 'Staph.', 'Strep.'], fontsize=14, style='italic')
ax.set_yticks(np.arange(9))
plt.title('True Concentration', fontsize=16)
plt.xlabel('Sample Number', fontsize=14)
plt.colorbar(shrink=0.7)
plt.xticks(np.arange(18), labels=np.arange(18)+1)
plt.show()


# true rel abudannce vs relative error
# Assume Ds[0] = 1
from matplotlib import cm
cmap = cm.get_cmap('hot')
lamTruePresent = lamTrue[lamTrue>0]
lamRecPresent = lamRec[:,:,0][lamTrue>0] #assess only bact actually in the sample
relErr = np.divide(np.abs(lamTruePresent - lamRecPresent), lamTruePresent)
relAbundance = lamTrue / np.sum(lamTrue,axis=0)
relAbPresent = relAbundance[lamTrue>0]
relAbRec = lamRec[:,:,0] / np.sum(lamRec[:,:,0],axis=0)
relAbRecPresent = relAbRec[lamTrue>0]
relRelErr = np.divide(np.abs(relAbPresent - relAbRecPresent), relAbPresent)
plt.figure(dpi=400)
for i in range(np.size(relRelErr)): 
    frac = lamTruePresent[i] / cmax
    plt.plot(relAbPresent[i], relRelErr[i], 'o', color=cmap(frac), markeredgecolor='k', markeredgewidth=mew, markersize=ms)

plt.xlabel('True Relative Abundance', fontsize=fs)
plt.ylabel('Relative Error', fontsize=fs)
ax = plt.axes()
ax.tick_params(axis='both', which='major', labelsize=fs-1)
plt.xticks(np.arange(9)/10, labels=np.arange(9)/10, fontsize=14)

plt.show()

# for s
plt.figure(dpi=400)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'Times New Roman'
plt.plot(estCopyCount.flatten(), lamCosSim.flatten(), 'o')
plt.xscale('log')
plt.ylim([0, 1])
ax = plt.axes()
ax.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel('Estimated 16S Copy Count', fontsize=fs)
plt.ylabel('Cosine Similarity', fontsize=fs)
         



# for only Ds=1
plt.figure(dpi=400)

lamBinary = lamTrue > 0
fpr, tpr, _= roc_curve(lamBinary[:,:5].flatten(), lamRec[:,:5,0].flatten())
plt.plot(fpr, tpr)
auck2 =np.round(auc(fpr,tpr), decimals=3)    
fpr, tpr, _= roc_curve(lamBinary[:,6:12].flatten(), lamRec[:,6:12,0].flatten())
plt.plot(fpr, tpr)
auck3 =np.round(auc(fpr,tpr), decimals=3)
fpr, tpr, _= roc_curve(lamBinary[:,12:].flatten(), lamRec[:,12:,0].flatten())
plt.plot(fpr, tpr)
auck4 =np.round(auc(fpr,tpr), decimals=3)

fpr, tpr, _= roc_curve(lamBinary.flatten(), lamRec[:,:,0].flatten())
plt.plot(fpr, tpr, 'k--')
aucOverall =np.round(auc(fpr,tpr), decimals=3)


plt.xlabel('False Positive Rate', fontsize=fs)
plt.ylabel('True Positive Rate', fontsize=fs)
plt.legend(['$k=2$, AUC={}'.format(auck2), '$k=3$, AUC={}'.format(auck3), '$k=4$, AUC={}'.format(auck4), 'Overall AUC={}'.format(aucOverall)], fontsize=12)
ax = plt.axes()
ax.tick_params(axis='both', which='major', labelsize=fs-1)
plt.show()
