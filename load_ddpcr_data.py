# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 07:50:12 2023

@author: pkk24
"""
from os import listdir
import pandas as pd
import pickle
import numpy as np

saveOverwrite = False
saveFile = 'processed_raw_data.pkl'

dataFolder = '.\\ddpcr_data\\binarized_test_samples\\'
filePieces = ['', '_preproc_well_threshold_line_22-07-13.csv']

activeCols = [1,2,3,4]
activeRows = 'CDEFGH'
groupBy = 'cols' # 'cols' or 'rows'
groups = [ [1,2,3,4] , [5,6,7,8], [9,10,11,12] ] # list of lists - can do multiple sets of groups for more samples
G = 4
allFiles = listdir(dataFolder)
allGroupInds = []
allY = []




if groupBy == 'rows': 
    sampleLoop = activeCols # samples in non-group direction
elif groupBy == 'cols': 
    sampleLoop = activeRows

for gg in range(len(groups)): 
    for i in range(len(sampleLoop)):             
        # this just organizes some basic metadata
        groupWells = []
        if groupBy == 'rows': 
            for g in range(G):                             
                if len(str(sampleLoop[i])) == 1: 
                    groupWells.append(groups[gg][g] + '0' + str(sampleLoop[i]))
                else: 
                    groupWells.append(groups[gg][g] + str(sampleLoop[i]))
        elif groupBy == 'cols': 
            for g in range(G):                             
                if len(str(groups[gg][g])) == 1: 
                    groupWells.append(sampleLoop[i] + '0' + str(groups[gg][g]))
                else: 
                    groupWells.append(sampleLoop[i] + str(groups[gg][g]))
                        
        # this gets pre-binarized measurements and group info 
        Ys =[]
        numDroplets = []
        for g in range(G): 
            
            Y = pd.read_csv(dataFolder+filePieces[0]+ groupWells[g] + filePieces[1]).to_numpy()[:, [0,1]]
            # read-in issues, this ensures boolean data
            Y[:,0] = Y[:,0] > 0
            Y[:,1] = Y[:,1] > 0                    
            
            Ys.append(Y)  
            numDroplets.append(np.shape(Y)[0])
            
        Y = (np.concatenate(Ys, axis=0)).T          
        
        indexTrack = 0
        group_indices = np.zeros(Y.shape[1]) 
        for g in range(G): 
            group_indices[indexTrack : (indexTrack+numDroplets[g]) ] = g
            indexTrack += numDroplets[g]
                     
        allGroupInds.append(group_indices)
        allY.append(Y)        
    
if saveOverwrite is True: 
    file = open(saveFile, 'wb')
    pickle.dump([allGroupInds, allY], file)
    file.close()
    