# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:11:49 2020

@author: pkk24
"""
import os
import pickle 
from matplotlib import pyplot as plt
from scipy.io import savemat
import pdb

import numpy as np 
from spore import spore, mmv_models, mmv_utils


## INPUTS 
mainSeed = 1

Mper = 2
N = 10
k = 3
D = 100
lamTot = 10
lamConst = 1
G = 2
allVars = np.logspace(-3,0,10)
varChannel = 0.01
#varChannel = allVars[0]
numTrials = 1
#saveFile = './20-11-26_rng1_phiUnif_Var{}_M{}_N{}_k{}_D{}_lamTot{}_G{}_50trials.pkl'.format(varChannel, Mper, N, k, D, lamTot, G)
saveFile = './21-03-04_rng1_phiUnif_Var{}_M{}_N{}_k{}_D{}_lamConst{}_G{}_50trials.pkl'.format(varChannel, Mper, N, k, D, lamConst, G)

signalModel = mmv_utils.MMVPConstantLambda(N, D, k, lamConst, initialSeed=mainSeed)
fwdModelParamGen = mmv_models.PhiUniform(Mper, N, G) 
measModel = mmv_models.LinearWithFixedGaussianNoise
groupModel = mmv_models.AutoSPoReFwdModelGroup

## Main 

# Generate (or later, load) signals: 
allX, allLam = signalModel.gen_trials(numTrials)

# Generate models and get measurements 
allPhi  = fwdModelParamGen.gen_trials(numTrials, seed=mainSeed) 
allY = np.zeros((Mper, D, numTrials))
allFwdModels = []
np.random.seed(mainSeed)
for t in range(numTrials):         
    fmsTemp = []
    for g in range(G): 
        fmsTemp.append(measModel(allPhi[:,:,g,t], varChannel * np.ones(Mper)))
    fm = groupModel(fmsTemp, D)
    
    allY[:,:,t] = fm.x2y(allX[:,:,t]) 
                                        
    allFwdModels.append(fm)


# SAVE 
simsData = [allX, allLam, allY, allFwdModels]

if os.path.exists(saveFile) is False: 
    with open(saveFile, 'wb') as file: 
        pickle.dump(simsData, file)

    # Save data to .mat for Matlab baselines
    matDict = {}
    matDict['allX'] = allX
    matDict['allLam'] = allLam
    matDict['allY'] = allY
    matDict['allPhi'] = allPhi
    allGroupIndices = np.zeros((numTrials, D))
    for i in range(numTrials): 
        allGroupIndices[i,:] = allFwdModels[i].group_indices + 1 # Matlab is 1-indexed
    matDict['allGroupIndices'] = allGroupIndices
    savemat(saveFile[:-3] + 'mat', matDict)
    
else: 
    print('output file already exists - won\'t overwrite')
    