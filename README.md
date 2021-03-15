# SPoRe
Sparse Poisson Recovery (SPoRe) 

We present a new compressed sensing framework for multiple measurements of sparse Poisson signals. We are primarily motivated by a suite of biosensing applications of microfluidics where analytes (such as whole cells or biomarkers) are captured in small volume partitions according to a Poisson distribution. We recover the sparse parameter vector of Poisson rates through maximum likelihood estimation with our novel Sparse Poisson Recovery(SPoRe) algorithm.  SPoRe uses batch stochastic gradient ascent enabled by Monte Carlo approximationsof otherwise intractable gradients.

## Publications
[1] P. K. Kota, D. LeJeune, R. A. Drezek, and R. G. Baraniuk, "Extreme Compressed Sensing of Poisson Rates from Multiple Measurements," 2021
-arXiv submission is currently processing and set to be announced March 17 0:00 GMT

## Instructions

We recommend starting with demo_script_groups.py to become familiar with the components of SPoRe. This file can be run immediately to generate simulations and evaluate SPoRe's recovery performance with desired parameter settings. Understanding this demo script will require digging into specific files in the 'spore' folder, which contains the functionality of SPoRe.

## spore/
In short: 
- mmv_utils.py: Generates Poisson signal matrices (X*)
- mmv_models.py: Defines models for the probabilistic mapping of signal-to-measurements (X*->Y)
- spore.py: SPoRe algorithm and class for Monte Carlo sampling functions

## Custom Models with mmv_models.py
Our initial exploration focuses on linear signal-to-measurement models, but SPoRe is modular for any model that defines the conditional probability p(y|x). Users may want to define a custom application-specific model for their measurements. Our implementations of the linear models with Gaussian noise should serve as a useful template for defining your own models. Briefly: 
- A child of the FwdModel class can be created and customized. 
- Use FwdModelGroup to define 'sensor groups' [1] each with different forward models.
- SPoReFwdModelGroup is useful with raw data when the group assignments to signals is uneven or variable. AutoSPoReFwdModelGroup is primarily useful for simulations to evenly distribute group assignments across signals. 



