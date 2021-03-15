import unittest

import numpy as np

from spore import spore, mmv_models, mmv_utils


class TestPoissonSampler(unittest.TestCase):

    def test_poisson_sampler(self):

        M, N, B, S = 7, 10, 391, 233
        lamdas = np.random.rand(N) + 1
        Y = np.random.randn(M, B)

        # test getting different samples
        poisson_sampler = spore.PoissonSampler(lamdas, sample_same=False)
        X = poisson_sampler.sample(Y, S)
        self.assertEqual(X.shape, (N, S, B))
        # sample means should be pretty close to true means
        self.assertTrue(np.allclose(np.mean(X, axis=(1, 2)), lamdas, rtol=1e-2, atol=0))

        # test getting the same samples
        poisson_sampler = spore.PoissonSampler(lamdas, sample_same=True)
        X = poisson_sampler.sample(Y, S)
        self.assertEqual(X.shape, (N, S, 1))
        # sample means should be somewhat close to true means
        self.assertTrue(np.allclose(np.mean(X, axis=(1, 2)), lamdas, rtol=0.2, atol=0))

class TestRecovery(unittest.TestCase):
    
    def test_spore_functions(self):
        N, D, k, lamTot, initSeed, numTrials = 20, 100, 3, 5, 1, 1
        Mper, G = 5, 3        
        varChannel = np.ones(Mper) * 1e-3
        S = 100
        maxIter = 100
        minLam = 1e-3
        
        signalSource = mmv_utils.MMVP(N, D, k, lamTot, initialSeed=initSeed)
        phiSource = mmv_models.PhiUniform(Mper, N, G)        
        measModel = mmv_models.LinearWithFixedGaussianNoise
        X, lamStar = signalSource.gen_trials(numTrials)                        
        allPhi = phiSource.gen_trials(numTrials, seed=initSeed)          
        self.assertEqual(X.shape, (N, D, numTrials))
        self.assertEqual(allPhi.shape, (Mper, N, G, numTrials))    
        
        # Instantiate SPoRe
        fm = mmv_models.AutoSPoReFwdModelGroup([measModel(allPhi[:, :, 0, 0], varChannel)], D)
        X = X[:,:,0]
        Y = fm.x2y(X)        
        sampler = spore.PoissonSampler(np.zeros(N), sample_same=True)         
        
        recAlg = spore.SPoRe(N, fm, sampler, conv_rel=-1, max_iter=maxIter) #negative conv_rel -> will definitely hit maxIter
        
        # Check log likelihood calculations. if we 'sample' with the actual X's, ll should be much higher than random sample
        pqRatio = recAlg.sampler.pq_ratio(X[:, :, None])
        pyx = fm.py_x_batch(Y[:, None, :], X[:, :, None], np.arange(D)) # (S, B) array where here, S and B = D
        self.assertEqual(pyx.shape, (D, D))
        self.assertEqual(pqRatio.shape, (D,1))
        pagg = pyx*pqRatio
        self.assertEqual(pagg.shape, (D, D))
        LL1 = recAlg.log_likelihood(pagg)
        
        sampler._lam = np.random.uniform(size=N) + 1e-1     
        X_bad = sampler.sample(Y, D) # D samples
        pqRatio = recAlg.sampler.pq_ratio(X_bad)
        pyx_bad = fm.py_x_batch(Y[:, None, :], X_bad, np.arange(D))
        pagg_bad = pyx_bad*pqRatio
        LL2 = recAlg.log_likelihood(pagg_bad)
        
        self.assertGreater(np.mean(pyx), np.mean(pyx_bad))
        self.assertGreater(LL1, LL2)
        
        # Test gradient 
        #self.assertTrue(np.all(sampler._lam == np.zeros(N)))
        grad_good = recAlg.gradient(X[:,:,None], lamStar[:,0]+minLam, pagg)
        # Nearly all incorrect lambdas should have negative gradient (say ~80% of them)
        self.assertTrue(np.sum(grad_good < 0) > (N-k)*0.8, msg='{0}'.format(grad_good))
        
        grad_bad = recAlg.gradient(X_bad, sampler._lam, pagg_bad)
        self.assertEqual(X_bad.shape, (N,D,1))
        # Nearly all incorrect lambdas should have negative gradient (say ~80% of them)
        self.assertTrue(np.sum(grad_bad < 0) > (N-k)*0.8, msg='{0}; {1}'.format(grad_bad, np.sum(pagg_bad, axis=0)))
        
        # Test basic recovery        
        lamRec, excl, lamHist, llHist = recAlg.recover(Y, S) 
        
        # Check shapes
        self.assertEqual(lamRec.shape, (N,))
        self.assertEqual(excl.shape, (D,))
        self.assertEqual(lamHist.shape, (N,maxIter))
        self.assertEqual(llHist.shape, (maxIter,))
        
        # last value in lamHist should be closer to lambdaStar (reasonably getting closer)
        self.assertGreater(np.linalg.norm(lamHist[:,0] - lamStar[:,0]), \
                           np.linalg.norm(lamHist[:,maxIter-1] - lamStar[:,0]))
        
        # log-likelihood should at least be trending up (even though noisy)        
        #self.assertGreater(np.mean(llHist[round(maxIter/2):]), np.mean(llHist[:round(maxIter/2)]))
        # technically have negative infinite log likelihoods on many iterations - throws stuff off
        
if __name__ == '__main__':
    unittest.main()
#python -m unittest discover