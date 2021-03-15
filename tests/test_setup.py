# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:46:16 2020

@author: pkk24
"""
import unittest

import numpy as np
import pickle
import os

from spore import mmv_models, mmv_utils


class TestMMVPsetup(unittest.TestCase):

    def test_signal_gen(self):
        N, D, k, lamTot, initSeed = 20, 1000, 3, 5,  1
        signalSource = mmv_utils.MMVP( N, D, k, lamTot, initialSeed=initSeed)
        
        numTrials = 100
        x1, l1 = signalSource.gen_trials(numTrials)
        x2, l2 = signalSource.gen_trials(numTrials, seed=2)
        
        # test new trials should be different if desired
        self.assertFalse(np.all(x1 == x2))
        self.assertFalse(np.all(l1 == l2))
        
        # Regenerate initial trials and check equality
        x1explicit, l1explicit = signalSource.gen_trials(numTrials, seed=signalSource.initialSeed)
        x1implicit, l1implicit = signalSource.gen_trials(numTrials)
        self.assertTrue(np.all(x1explicit==x1))
        self.assertTrue(np.all(l1explicit==l1))
        self.assertTrue(np.all(x1explicit==x1implicit))
        self.assertTrue(np.all(l1explicit==l1implicit))
        
        # Check shape
        self.assertEqual(x1.shape, (N, D, numTrials))
        self.assertEqual(l1.shape, (N,numTrials))
        
        # MLE estimates of lambda directly from X should be pretty close to true lambdas
        self.assertTrue(np.allclose(np.mean(x1, axis=1), l1, rtol=1e-2, atol=0.5))       
        
        # Test ability to generate trials, save, and reload
        xSave, lSave = signalSource.gen_trials(numTrials, savePath='unittest_saving.pickle')
        with open('unittest_saving.pickle', 'rb') as testfile: 
            signalLoad = pickle.load(testfile)
        self.assertTrue(np.all(xSave == signalLoad['allX']))
        self.assertTrue(np.all(lSave == signalLoad['allLambdaStars']))

        os.remove('unittest_saving.pickle')
        
    def test_linear_models(self):
        N, D, k, lamTot, initSeed, numTrials = 20, 1000, 3, 5, 1, 10
        Mper, G = 5, 3
        
        signalSource = mmv_utils.MMVP(N, D, k, lamTot, initialSeed=initSeed)
        phiSource = mmv_models.PhiUniform(Mper, N, D, G)        
        # get signals
        allX, allLam = signalSource.gen_trials(numTrials)        
        
        
        # get sensing matrices
        allPhi, groupIndices = phiSource.gen_trials(numTrials, seed=initSeed)                
        # Verify seeding is working
        allPhi2, _ = phiSource.gen_trials(numTrials, seed=initSeed)        
        self.assertTrue(np.all(allPhi2 == allPhi))
        allPhi3 = phiSource.gen_trials(numTrials)            
        
        #why is this getting a deprecation warning but the other assertFalses don't?     
        self.assertFalse(np.all(allPhi3 == allPhi2)) # not seeded -> different
        #Comment this out and no warning            
            
        self.assertFalse(np.all(allPhi[:,:,0,0] == allPhi[:,:,-1,0])) # splits => different matrices

        # Verify splitting from stacked phi is working as intended        
        phiSource2 = mmv_models.PhiUniform(Mper*G, N, D, G=1)
        bigPhi,bigGroup = phiSource2.gen_trials(numTrials, seed=initSeed)
        
        self.assertTrue(np.all(bigGroup == np.zeros(D,))) # one group if G = 1
        # First few sensors in stacked matrix equal group 1 matrices
        self.assertTrue(np.all(bigPhi[:Mper,:,0,:] == allPhi[:,:,0,:]))
        # Last sensor = last sensor
        self.assertTrue(np.all(bigPhi[-1,:,0,:] == allPhi[-1,:,-1,:]))
        # Shapes
        self.assertEqual(bigPhi.shape, (Mper*G, N, D, numTrials))
        self.assertEqual(allPhi.shape, (Mper, N, D, numTrials))
        
        # Test different measurement models            
        varChannel = 1e-5
        varChannelvec = np.ones(Mper*G)*varChannel
        fwdmodel = mmv_models.LinearGaussianFixedDiagonalCovariance(bigPhi[:,:,:,0], bigGroup, varChannelvec)        
        Y = fwdmodel.x2y(allX[:,:,0], seed=1)
        
        # Test seeding
        Y2 = fwdmodel.x2y(allX[:,:,0])
        Y3 = fwdmodel.x2y(allX[:,:,0], seed=1)
        self.assertTrue(np.all(Y != Y2))
        self.assertTrue(np.all(Y == Y3))
        
        # Test shape
        self.assertEqual(Y.shape, (Mper*G, D,))        
        # y be reasonably close to phi x for small Sigma        
        self.assertTrue(np.allclose(Y, bigPhi[:,:,0,0] @ allX[:,:,0], rtol=1e-1, atol=0.1))

                         
        # Covariances should be equal
        Sigmas0 = fwdmodel.get_sigmas(allX[:,:,0])
        Sigmas1 = fwdmodel.get_sigmas(allX[:,:,1])
        # within a set of X's - arbitrarily test first and last
        self.assertTrue(np.all(Sigmas0[:,:,0] == Sigmas0[:,:,-1]))
        # and between a set of X's
        self.assertTrue(np.all(Sigmas0[:,:,0] == Sigmas1[:,:,-1]))
        

        # Test scaling variance
        fwdmodel = mmv_models.LinearWithScalingVarianceGaussianNoise(bigPhi[:,:,:,0], bigGroup, varChannelvec, varMag=1e-2)
        Sigmas = fwdmodel.get_sigmas(allX[:,:,0])
        # Covariances should not be equal
        self.assertFalse(np.all(Sigmas[:,:,0] == Sigmas[:,:,-1]))

        # Test G > 1, quick size check        
        varChannelvec = np.ones(Mper)*varChannel
        fwdmodel = mmv_models.LinearGaussianFixedDiagonalCovariance(allPhi[:,:,:,0], groupIndices, varChannelvec)            
        Y = fwdmodel.x2y(allX[:,:,0], seed=1)
        self.assertEqual(Y.shape, (Mper, D,))        

        # still need test for py_x


if __name__ == '__main__':
    unittest.main()
