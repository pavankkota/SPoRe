# -*- coding: utf-8 -*-
"""
Generate final plots involving simulated data (flagging out of panel barcodes, and supp figure on performance)

[1] Kota et al, Expanded Multiplexing on Sensor-Constrained Microfluidic Partitioning Systems
    https://www.biorxiv.org/content/10.1101/2022.12.23.521805v1
    
@author: pkk24
"""

import pickle
from os import listdir
import pdb
import pandas as pd
import numpy as np
from spore import mmvp_exact_grad, mmv_models, mmv_utils
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
convTol = 1e-8 #note this was 1e-6 in the original bioRxiv post, but it was found that lambda wasn't fully converging
p_err = 0
epsilon = 1e-6
groupWeight=True # proportional weighting to groups in optimization.  if True, scale by droplet count
lamTrue2 = deepcopy(lamTrue)
lamTrue2[lamTrue2==0] = epsilon

#chi sq stuff
flagTest = True # if evaluating the ability to flag out-of-panel barcodes. Increases total runtime
ddof=0



## model for simulating
BCprobs = np.zeros((4, 9, G))

BCprobs[:,:,0] = np.array([[0, 0, 0,     0, 0, 0, 0, 1, 0], \
                           [1, 0, 0.125, 0, 1, 0, 1, 0, 1], \
                           [0, 1, 0,     1, 0, 0, 0, 0, 0], \
                           [0, 0, 0.875, 0, 0, 1, 0, 0, 0] ])

BCprobs[:,:,1] = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0], \
                           [1, 0, 1, 0, 1, 1, 0, 0, 1], \
                           [0, 0, 0, 0, 0, 0, 0, 1, 0], \
                           [0, 0, 0, 0, 0, 0, 1, 0, 0] ])
    
BCprobs[:,:,2] = np.array([[0, 0, 0,     0, 1, 0, 0, 1, 1], \
                           [1, 0, 0.125, 0, 0, 0, 1, 0, 0], \
                           [0, 1, 0,     1, 0, 1, 0, 0, 0], \
                           [0, 0, 0.875, 0, 0, 0, 0, 0, 0] ])

BCprobs[:,:,3] = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0], \
                           [1, 0, 1, 0, 1, 1, 1, 0, 0], \
                           [0, 0, 0, 1, 0, 0, 0, 1, 0], \
                           [0, 0, 0, 0, 0, 0, 0, 0, 1] ])


BCs = np.array([[0, 0, 1, 1], \
                [0, 1, 0, 1]])
    
if np.any(np.sum(BCprobs, axis=0) != 1): 
    raise ValueError('Check BCprobs')

gModels = []
for g in range(G): 
    gModels.append(mmv_models.BooleanYwithMixtureX(BCs, BCprobs[:,:,g]))    
groupModel = mmv_models.FwdModelGroup(gModels)


"""
Main
"""
truePvals = []
badPvals = []
domPvals = []
pctOmit = []
truePvalsSim = []
badPvalsSim = []

allFMG = []
sporeObjs = []

lamRec = np.zeros( (N, numSamples))
lamRecSim = np.zeros( (N, numSamples))
nllCompare = np.zeros((numSamples, 2))
lamCosSim = np.zeros((numSamples))
lamCosSimSim = np.zeros((numSamples))


np.random.seed(mainSeed)
for i in range(numSamples): 
#for i in range(5): 
    mleObj = mmvp_exact_grad.BinaryMLE(PSI, p_err)
    


    lam0 = np.ones((N,))*0.1
    lamRec[:,i] = mleObj.recover(allY[i], allGroupInds[i], lam0, groupWeight=groupWeight, convtol=convTol)

    # Simulate data from lamTrue
    simModel = mmv_models.SPoReFwdModelGroup(groupModel, allGroupInds[i])
    sigModel = mmv_utils.MMVPInputLambda(allY[i].shape[1], lamTrue[:,i])
    simX,_ = sigModel.xgen()
    simY = simModel.x2y(simX)
    
    mleObjSim = mmvp_exact_grad.BinaryMLE(PSI, p_err)
    lam0 = np.ones((N,))*0.1
    lamRecSim[:,i] = mleObjSim.recover(simY, allGroupInds[i], lam0, groupWeight=groupWeight, convtol=convTol)
    lamCosSimSim[i] = 1-cosine(lamRecSim[:,i], lamTrue[:,i])
    lamCosSim[i] = 1-cosine(lamRec[:,i], lamTrue[:,i])
    print(lamCosSim[i])
    print(lamCosSimSim[i])

    groupWeights = np.zeros(G)
    for g in range(G):
        groupWeights[g] = np.sum(allGroupInds[i]==g) / np.size(allGroupInds[i])
        
    pYempirical = mleObj.get_p_y(allY[i], allGroupInds[i])        
    nllCompare[i,0] = mleObj.NLL(mleObj.lam_reduce(lamRec[:,i]), allY[i], groupWeights, pYempirical)    
    nllCompare[i,1] = mleObj.NLL(mleObj.lam_reduce(lamTrue2[:,i]), allY[i], groupWeights, pYempirical)
    
    

    
    #Now want E[-log p(y|\lambda_rec)], var[-log p(y|\lambda_rec)] versus the empirical distribution
    lamRecReduced = mleObj.lam_reduce(lamRec[:,i])
    
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
    

    # copy and do the same thing for simulated data
    lamRecReducedSim = mleObjSim.lam_reduce(lamRecSim[:,i])
    pYempiricalSim = mleObjSim.get_p_y(simY, allGroupInds[i])        

    nllYSim = np.zeros((G,4))    
    for g in range(G): 
        nllYSim[g,:] = mleObjSim.nlog_py_lam(lamRecReducedSim[g,:])    
        #pY[g,:] = mleObj.py_lam(lamRecReduced[g,:])
    
    pYSim = np.exp(-nllYSim)
    pYSim = pYSim * groupWeights[:,None] #actual expected distribution of Y's given the # of droplets in each group
    nllMean = np.sum(np.multiply(pY, nllY))
    nllVar = np.sum(np.multiply(pY, (nllY - nllMean)**2))

    #Empirical:         
    pYempiricalAdjustSim = pYempiricalSim.T*groupWeights[:,None]
    nllMeanEmpirical = nllCompare[i,0]
    nllVarEmpirical = np.sum(np.multiply(pYempiricalAdjust, (nllY - nllMeanEmpirical)**2))
    
    chival, pval = chisquare(pYempiricalAdjustSim * simY.shape[1], pYSim*simY.shape[1], ddof=ddof, axis=None)
    truePvalsSim.append(pval)
  
    
    if flagTest:        
        supp = np.where(lamTrue[:,i] !=0)[0]        
        for k in range(len(supp)): 
            #lamTemp = np.delete(lamTrue[:,i], supp[k])
            PSItemp = np.delete(PSI, supp[k], axis=1)
            lam0temp = np.delete(lam0, supp[k])
            mleObj = mmvp_exact_grad.BinaryMLE(PSItemp, p_err)
            lamRecTemp = mleObj.recover(allY[i], allGroupInds[i], lam0temp, groupWeight=groupWeight, convtol=convTol)
            
            expNLL, empNLL, chisqinfo = dist_check(mleObj, lamRecTemp, groupWeights, allY[i].shape[1], ddof=ddof)
            badPvals.append(chisqinfo[1])
            pctOmit.append(lamTrue[supp[k], i] / np.sum(lamTrue[:,i]))
            
            if lamTrue[supp[k],i] == np.max(lamTrue[:,i]):
                domPvals.append(chisqinfo[1])
                
            mleObjSim = mmvp_exact_grad.BinaryMLE(PSItemp, p_err)
            lamRecSimTemp = mleObj.recover(simY, allGroupInds[i], lam0temp, groupWeight=groupWeight, convtol=convTol)
            expNLL, empNLL, chisqinfo = dist_check(mleObjSim, lamRecSimTemp, groupWeights, simY.shape[1], ddof=ddof)

            badPvalsSim.append(chisqinfo[1])
            
         
        
    print('Iteration ' + str(i+1) + '/' + str(numSamples) + ' complete')
    
lamCosSim2 = np.zeros(numSamples)
for i in range(numSamples): 
    lamCosSim[i] = 1-cosine(lamRec[:,i], lamTrue[:,i])
    lamCosSim2[i] = 1-cosine(lamRec[:,i], lamTrue2[:,i])





plt.figure()
plt.imshow(lamRec, cmap='hot', vmin=0, vmax=2.5568)
plt.title('Recovered Concentration with SPoRe', fontsize=16)
ax = plt.gca()
ax.set_yticklabels(['A. baum.', 'B. frag.', 'E. cloac.', 'E. faec.', 'E. coli', 'K. pneu', \
               'P. aeru', 'Staph.', 'Strep.'], fontsize=14)
ax.set_yticks(np.arange(9))
plt.xlabel('Sample Number', fontsize=14)
plt.xticks(np.arange(18), labels=np.arange(18)+1)

plt.figure()
plt.imshow(lamRecSim, cmap='hot', vmin=0, vmax=2.5568)
plt.title('Recovered Concentration with SPoRe - Simulated Data', fontsize=16)
ax = plt.gca()
ax.set_yticklabels(['A. baum.', 'B. frag.', 'E. cloac.', 'E. faec.', 'E. coli', 'K. pneu', \
               'P. aeru', 'Staph.', 'Strep.'], fontsize=14)
ax.set_yticks(np.arange(9))
plt.xlabel('Sample Number', fontsize=14)
plt.xticks(np.arange(18), labels=np.arange(18)+1)


plt.show()

plt.figure()
plt.imshow(lamTrue, cmap='hot', vmin=0, vmax=2.5568)
ax = plt.gca()
ax.set_yticklabels(['A. baum.', 'B. frag.', 'E. cloac.', 'E. faec.', 'E. coli', 'K. pneu', \
               'P. aeru', 'Staph.', 'Strep.'], fontsize=14)
ax.set_yticks(np.arange(9))
plt.title('True Concentration', fontsize=16)
plt.xlabel('Sample Number', fontsize=14)
plt.xticks(np.arange(18), labels=np.arange(18)+1)

plt.show()


plt.figure(dpi=400)
lamTruePresent = lamTrue[lamTrue>0]
lamRecSimPresent = lamRecSim[lamTrue>0] #assess only bact actually in the sample
relErr = np.divide(np.abs(lamTruePresent - lamRecSimPresent), lamTruePresent)

#relAbundance = lamTrue / np.sum(lamTrue,axis=0)
#relAbPresent = relAbundance[lamTrue>0]
plt.plot(lamTruePresent, relErr, 'o')
plt.xlabel('True Absolute Abundance ($\lambda_n$)', fontsize=14)
plt.ylabel('Relative Error', fontsize=14)
plt.ylim([0,1])
ax = plt.axes()
ax.tick_params(axis='both', which='major', labelsize=13)
plt.show()

if flagTest: 
    plt.figure()
    truePvals = np.array(truePvals)
    badPvals = np.array(badPvals)
    pctOmit = np.array(pctOmit)
    
    from matplotlib import pyplot as plt
    from sklearn.metrics import roc_curve, auc
    alpha = 0.5
    #bins = np.hstack((np.array([0]), np.logspace(-300, 0, 25)))
    truePvals2 = np.log10(truePvals)
    badPvals2 = np.log10(badPvals)
    allvals = np.hstack((truePvals2, badPvals2))
    minVal = np.nanmin(allvals[allvals!=-np.inf])
    truePvals2[np.isinf(truePvals2)] = minVal
    badPvals2[np.isinf(badPvals2)] = minVal
    
    plt.hist(truePvals2, alpha=0.8, label='True p', color='r')
    plt.hist(badPvals2,  alpha=0.8, label='Missing p', color='b')
    plt.legend(loc='upper right')
    plt.show()
    
    plt.figure()
    plt.hist(pctOmit[badPvals==0], alpha=0.8, label='Flagged (p=0)', color='r')
    plt.hist(pctOmit[badPvals!=0],  alpha=0.8, label='Unflagged (p>0)', color='b')
    plt.title('Flagging Unknown Microbes')
    plt.xlabel('Percent of Sample')
    plt.legend(loc='upper right')
    plt.show()
    
    flagLabels = np.hstack( (np.zeros(truePvals.size), np.ones(badPvals.size)) )
    
    #negate p values such that small p values become large (closer to 0-) for 'positive' label (a 'flag')
    plt.figure(dpi=400)
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.family'] = 'Times New Roman'
    fpr, tpr, _= roc_curve(flagLabels, -np.hstack((truePvals, badPvals)))
    plt.plot(fpr, tpr)
    fprSim, tprSim, _= roc_curve(flagLabels, -np.hstack((truePvalsSim, badPvalsSim)))
    plt.plot(fprSim, tprSim)
    plt.legend(['Manual Thresholded Measurements: AUC={}'.format(np.round(auc(fpr,tpr), decimals=3)), \
                'Simulated Measurements: AUC={}'.format(np.round(auc(fprSim,tprSim), decimals=3))])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    ax = plt.axes()
    ax.tick_params(axis='both', which='major', labelsize=13)
    #plt.title('Labeling Samples with Unknown Barcodes: ROC-AUC={}'.format()

         
