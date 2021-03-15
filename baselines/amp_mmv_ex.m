clear yAMP1 phiAMP1
D = 100; 
M = 5; 
N = 20; 
k = 3;
activeMean = 1; 
activeVar = 1; 
%RunOpt = Options();
%RunOpt.verbose = false; 
RunOpt = Options('smooth_iters', 100, 'inner_iters', 25, 'verbose', false);
% Can try with and without this artificial ~temporal correlation
[~, sortInds] = sort(vecnorm(Y(:,1:D))); 
Y2 = Y(:,sortInds); 

for i = 1:D
yAMP1(i) = {Y2(:,i)};
phiAMP1(i) = {phi};
end
SigGenObj = SigGenParams('M', M, 'N', N, 'T', D, 'version', 'R', 'zeta', activeMean, 'sigma2', activeVar, 'lambda', k/N);     % Call the default constructor
%[x_true, y, A, support, K, sig2e] = signal_gen_fxn(SigGenObj);
Params = ModelParams(SigGenObj,varFlat);
[x_hat, v_hat, lambda_hat] = sp_mmv_wrapper_fxn(yAMP1, phiAMP1, Params, RunOpt);

[sum([x_hat{:}],2), sum(X(:,1:D),2)]