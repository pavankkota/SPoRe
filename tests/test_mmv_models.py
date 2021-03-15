from itertools import product
import unittest

import numpy as np
from scipy.stats import multivariate_normal as mvn

from spore import mmv_models


class TestFunctions(unittest.TestCase):

    def setUp(self):

        self.M = 5
        self.S = 4
        self.B = 6

        self.mu = np.random.rand(self.M, self.S, 1)

        self.Sigma_sqrt_full = np.random.rand(self.M, self.M, 1, self.B)
        self.Sigma_full = np.einsum('ijlm,kjlm->iklm', self.Sigma_sqrt_full, self.Sigma_sqrt_full)
        self.Sigma_inv_full = np.zeros_like(self.Sigma_full)
        self.Sigma_det_full = np.zeros((1, self.B))
        for i in range(self.B):
            self.Sigma_inv_full[:, :, 0, i] = np.linalg.inv(self.Sigma_full[:, :, 0, i])
            self.Sigma_det_full[0, i] = np.linalg.det(self.Sigma_full[:, :, 0, i])
        
        # get Sigma_diag from the diagonals of Sigma_full
        self.Sigma_diag = self.Sigma_full[np.arange(self.M), np.arange(self.M), :, :]
        self.Sigma_sqrt_diag = np.sqrt(self.Sigma_diag)
        self.Sigma_inv_diag = 1 / self.Sigma_diag
        self.Sigma_det_diag = np.prod(self.Sigma_diag, axis=0)

    def test_invert_det_sqrtm_batch(self):

        # try 2d matrix
        A = self.Sigma_full[:, :, 0, 0]
        A_inv = np.linalg.inv(A)
        A_det = np.linalg.det(A)

        A_inv_2, A_det_2 = mmv_models._invert_det_batch(A)
        A_sqrt_2 = mmv_models._sqrtm_batch(A)

        self.assertTrue(np.allclose(A_inv, A_inv_2))
        self.assertTrue(np.allclose(A_det, A_det_2))
        self.assertTrue(np.allclose(A, A_sqrt_2 @ A_sqrt_2))

        # try higher-order tensor
        Sigma_inv, Sigma_det = mmv_models._invert_det_batch(self.Sigma_full)
        Sigma_sqrt = mmv_models._sqrtm_batch(self.Sigma_full)
        Sigma = np.einsum('ij...,jk...->ik...', Sigma_sqrt, Sigma_sqrt)

        self.assertTrue(np.allclose(self.Sigma_inv_full, Sigma_inv))
        self.assertTrue(np.allclose(self.Sigma_det_full, Sigma_det))
        self.assertTrue(np.allclose(self.Sigma_full, Sigma))

    def test_diag(self):

        # test going from vectors to diagonal matrices
        Sigma_full = mmv_models._diag(self.Sigma_diag)

        for j in range(self.B):
            self.assertTrue(np.allclose(np.diag(np.diag(self.Sigma_full[:, :, 0, j])), Sigma_full[:, :, 0, j]))
    
    def test_gaussian_sample_and_pdf(self):

        # test broadcasting with diagonal covariance
        X = mmv_models._gaussian_sample(self.mu, self.Sigma_sqrt_full)
        self.assertTupleEqual(X.shape, (self.M, self.S, self.B))

        # test broadcasting with full covariance
        X = mmv_models._gaussian_sample(self.mu, self.Sigma_sqrt_diag)
        self.assertTupleEqual(X.shape, (self.M, self.S, self.B))
    
    def test_gaussian_pdf(self):

        X = np.random.randn(self.M, self.S, self.B)

        # test broadcasting with diagonal covariance
        pdfs = np.zeros((self.S, self.B))
        for i, j in product(range(self.S), range(self.B)):
            pdfs[i, j] = mvn.pdf(X[:, i, j], self.mu[:, i, 0], np.diag(self.Sigma_diag[:, 0, j]))
        pdfs_2 = mmv_models._gaussian_pdf(X, self.mu, self.Sigma_inv_diag, self.Sigma_det_diag)

        self.assertTrue(np.allclose(pdfs, pdfs_2))

        # test broadcasting with full covariance
        for i, j in product(range(self.S), range(self.B)):
            pdfs[i, j] = mvn.pdf(X[:, i, j], self.mu[:, i, 0], self.Sigma_full[:, :, 0, j])
        pdfs_2 = mmv_models._gaussian_pdf(X, self.mu, self.Sigma_inv_full, self.Sigma_det_full)

        self.assertTrue(np.allclose(pdfs, pdfs_2))


if __name__ == '__main__':
    unittest.main()