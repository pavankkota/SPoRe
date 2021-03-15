# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:11:49 2020

@author: pkk24

Authors: Pavan Kota, Daniel LeJeune

Reference: 
P. K. Kota, D. LeJeune, R. A. Drezek, and R. G. Baraniuk, "Extreme Compressed 
Sensing of Poisson Rates from Multiple Measurements," Mar. 2021.

arXiv ID:
"""

import cProfile
from matplotlib import pyplot as plt

import numpy as np 
from spore import spore, mmv_models, mmv_utils

# Really basic setup script
Mper = 2
N = 10
G = 2
D = 1000
k = 2
lamTot = 6
mainSeed = 1
S = 1000
numTrials = 1
maxIter=10000
profiling = False 

signalModel = mmv_utils.MMVP(N, D, k, lamTot, initialSeed=mainSeed)
fwdModelParamGen = mmv_models.PhiUniform(Mper, N, G) 
measModel = mmv_models.LinearWithFixedGaussianNoise
groupModel = mmv_models.AutoSPoReFwdModelGroup
varChannel = 1e-2

# Generate (or later, load) signals: 
allX, allLam = signalModel.gen_trials(numTrials)

# Generate models and get measurements (or later, load them): 
allPhi = fwdModelParamGen.gen_trials(numTrials, seed=mainSeed) # allPhi is (M, N, G, numTrials)

# Define models and simulate measurements
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


# Recover
sampler = spore.PoissonSampler(np.zeros(N), sample_same=True)
lamS = np.zeros((N, numTrials))
def run_trials(): 
    for t in range(numTrials):     
        recAlg = spore.SPoRe(N, allFwdModels[t], sampler, conv_window=500, conv_rel=0.05, max_iter=maxIter, step_size=1e-2)
        lamS[:,t], included, lamHist, llHist = recAlg.recover(allY[:,:,t], S)        
        cossim = np.dot(lamS[:,t], allLam[:,t]) / (np.linalg.norm(lamS[:,t])*np.linalg.norm(allLam[:,t]))
        print('Trial ' + str(t+1) + ', cosine similarity: ' + str(cossim))

if profiling: 
    def run():
        run_trials()
    cProfile.run('run()', sort='tottime')    
else: 
    run_trials()    