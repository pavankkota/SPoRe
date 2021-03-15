# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:54:56 2021

@author: pkk24
"""

import numpy as np 
import sys
import matplotlib.pyplot as plt
import pickle
import os

from matplotlib import cm
from scipy.stats import poisson
from itertools import product
from copy import deepcopy
from scipy.io import savemat

newpath = "../../"
if newpath not in sys.path:
    sys.path.append(newpath)
from spore import spore, mmv_models, mmv_utils

import pdb

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


mainSeed = 1
Mper = 1
D = 1000
#varChannel = 0.01
varChannel = 0.03
lamVec = np.array([0.5, 0, 0.5])
N = np.size(lamVec)

if Mper == 1: 
    phi = np.array([[1, 2, 3]])
elif Mper == 2:
    phi = np.array([[0, 1, 1], [1, 0, 1]])
    m2_3d = True

phi = phi[:,:,None]
Mper, _, G = phi.shape

signalModel = mmv_utils.MMVPInputLambda(D, lamVec, initialSeed=mainSeed)


measModel = mmv_models.LinearWithFixedGaussianNoise
groupModel = mmv_models.AutoSPoReFwdModelGroup

## Main 

# Generate (or later, load) signals: 
allX, allLam = signalModel.gen_trials(1)

# Generate models and get measurements 
allY = np.zeros((Mper, D, 1))
allFwdModels = []
np.random.seed(mainSeed)

fmsTemp = []
for g in range(G): 
    fmsTemp.append(measModel(phi[:,:,g], varChannel * np.ones(Mper)))
fm = groupModel(fmsTemp, D)

Y = fm.x2y(allX[:,:,0])

# Recover with SPoRe
S = 1000
maxIterSPoRe=100000

sporeObjs = []
Ys = []
sampler = spore.PoissonSampler(np.zeros(N),sample_same=True)

sporeObj = spore.SPoRe(N, fm, sampler, max_iter=maxIterSPoRe)

lamSPoRe,_,_,_ = sporeObj.recover(Y, S)


#Get l1-oracle solution from matlab
saveFile = 'mixfig_matlab.mat'
if os.path.exists(saveFile) is False: 
    matDict = {}
    matDict['allX'] = allX
    matDict['allLam'] = allLam
    matDict['allY'] = Y
    matDict['allPhi'] = phi
    matDict['allGroupIndices'] = fm.group_indices + 1 # Matlab is 1-indexed
    savemat(saveFile, matDict)
    
xMax = np.max(allX)+3
lam0 = np.ones(N)*(1e-1) # used in spore.recover by default

if Mper == 1: 
    yrange = np.linspace(-1, np.ceil(np.max(Y)), 1000)
    yrange = yrange[None,None,:]
    plt.figure(dpi=400)
    # Plot
    #plt.plot(Y, np.zeros(Y.shape), 'ko')
    plt.hist(Y[0,:], 100, density=True, color=np.array([1,1,1])*0.65, label=r'$D$ Observations')
    plt.plot(yrange[0,0,:], py_lam(fm, yrange, xMax, N, lamVec), '-', color=np.array([1,1,1])*0, linewidth=1.5, label='True Distribution')
    plt.plot(yrange[0,0,:], py_lam(fm, yrange, xMax, N, lamSPoRe), color='b', linestyle=(0, (1,1)), linewidth=4, label='SPoRe')
    plt.plot(yrange[0,0,:], py_lam(fm, yrange, xMax, N, np.array([0.3349, 0.3069, 0.3082])), color='r', linestyle=(0, (1,1)), label=r'$\ell_1$-Oracle MMV')
    #plt.plot(yrange[0,0,:], py_lam(fm, yrange, xMax, N, lam0), color=np.array([1,1,1])*0.75)
    #plt.ylim((0, 2))
    font = {'family': 'serif',        
            'weight': 'normal',     
            'size': 12
            }
    axLabelSize=13
    axTickSize=13
    titleSize=14
    
    #plt.legend(['True Distribution', 'SPoRe', r'$l_1$-Oracle', r'$D$ Observations'],prop=font)
    handles,labels = plt.gca().get_legend_handles_labels()
    #labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    #plt.legend(handles, labels)
    order = [0,3,1,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], prop=font)

    #A,B = plt.gca().get_legend_handles_labels()

    plt.xlabel(r'$y$', size=axLabelSize)
    plt.ylabel(r'$p(y|\mathbf{\lambda})$', size=axLabelSize)
    
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    ax = plt.axes()
    ax.tick_params(axis='both', which='major', labelsize=axTickSize)

    for tick in ax.get_xticklabels():
        tick.set_fontname("DejaVu Serif")
    for tick in ax.get_yticklabels():
        tick.set_fontname("DejaVu Serif")
    #plt.legend([r'$p(y|\mathbf{\lambda}*)$', r'$p(y|\widehat{\mathbf{\lambda}})$', r'$p(y|\hat{\mathbf{\lambda}}_{l1})$'])

elif Mper == 2: 
    n = 250
    yrange = np.linspace(-1, np.ceil(np.max(Y)), n)    
    y_i = [yrange]*Mper
    yRange = np.array(list(product(*y_i))).T #     
    yRange = yRange[:, None, :]
    Z = py_lam(fm, yRange, xMax, N, lamVec)
    Z2 = py_lam(fm, yRange, xMax, N, np.array([0.3,0.3,0.3]))

    if m2_3d:         
        Xm,Ym = np.meshgrid(yrange, yrange)
        fig = plt.figure(dpi=400)
        ax = fig.gca()
        
        ax = fig.gca(projection='3d')
        #ax.plot_surface(Xm, Ym, np.reshape(Z, (n,n)), cmap=cm.Blues, alpha = 0.8, vmin=-1, zorder=-10)
        #ax.plot_wireframe(Xm, Ym, np.reshape(Z, (n,n)), linewidth=0.5, edgecolor='k', alpha=0.5)
        ax.view_init(elev=50, azim=-60)
        
        ax.plot_surface(Xm, Ym, np.reshape(Z2, (n,n)), cmap=cm.Reds, alpha = 0.8, vmin=-1, zorder=-10)
        ax.scatter(Y[0,:], Y[1,:], np.zeros(D), s=1)        

        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # ax.contourf(Xm, Ym, np.reshape(Z, (n,n)), zorder=-1)        
        # ax.scatter(Y[0,:], Y[1,:], s=1, zorder=1)
        #plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        ax.set_xlabel(r'$y_1$',size=10)
        ax.set_ylabel(r'$y_2$', size=10)
        ax.set_zlabel(r'$p(y|\lambda)$')
        #ax = plt.axes()
        for tick in ax.get_xticklabels():
            tick.set_fontname("DejaVu Serif")
        for tick in ax.get_yticklabels():
            tick.set_fontname("DejaVu Serif")
        ax.set_zticks([])
        ax.dist=13
        
    else: 
        plt.imshow(np.reshape(Z, (n,n)), alpha=0.75, cmap='hot')
        plt.plot(Y[0,:], Y[1,:], 'ko')