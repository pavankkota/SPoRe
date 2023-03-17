"""
This script mainly is used to estimate E cloacae's partial cluster (i.e., proportion
of its 16S genes that probe 1 binds to). Used it to store other bacterial estimated
ground truths as well
"""

from os import listdir
import pandas as pd
import numpy as np
from spore import mmvp_exact_grad



G = 1
singleBactCols = np.arange(12)+1
groupRows = 'AB'
dataFolder = '.\\ddpcr_data\\binarized_reference_data\\'
filePieces = ['', '_preproc_well_threshold.csv']

dfColNames = ['Well ID', 'Lambda']
finalData = pd.DataFrame(columns=dfColNames)



allFiles = listdir(dataFolder)
ecloac_allprobes = []
ecloac_p = []
ecloacAll = []
ecloacPartial = []

for i in range(len(singleBactCols)): 
    for j in range(len(groupRows)): 
        if len(str(singleBactCols[i])) == 1:
            wellID = groupRows[j] + '0' + str(singleBactCols[i])
        else:
            wellID = groupRows[j] + str(singleBactCols[i])
        
        
        # switch FAM and HEX ordering to match how they plot it in Bio-Rad's software
        Ys = []        
        Ys=((pd.read_csv(dataFolder+filePieces[0]+ wellID + filePieces[1])).to_numpy()[:, [1,0]])
        
        negPortion = np.sum(np.logical_and(Ys[:,0]==0, Ys[:,1] == 0)) / Ys.shape[0]
        
        lamEst = -np.log(negPortion)
        
        if wellID == 'A03' or wellID == 'B10':  #E cloac
            
            group_inds = np.zeros(Ys.shape[0])            
            
            PSI = np.array([[0, 1, 1], [1, 1, 0]])[:,:,None]

            mleObj = mmvp_exact_grad.BinaryMLE(PSI)
            lamPartial, lamAll, lamHEXjunk = mleObj.recover(Ys.T, group_inds, np.ones(3)*0.1)
            ecloacAll.append(lamAll)
            ecloacPartial.append(lamPartial)
                        
        print('Well ID: ' + str(wellID) + ' complete')
        finalData = finalData.append(pd.DataFrame( [[wellID, lamEst]], columns=dfColNames))

ecloac_p = np.mean(np.array(ecloacAll)/(np.array(ecloacAll)+np.array(ecloacPartial)))
