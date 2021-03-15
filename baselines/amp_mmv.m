%Pavan Kota
%Created: August 23, 2020

%AMP-MMV version 1.1
%http://www2.ece.ohio-state.edu/~schniter/MMV/index.html

function X = amp_mmv(Y, Phis, group_indices, AWGN, kGuess) 
%kGuess optional? or necessary to initialize lambda (as defined in AMP-MMV
%- the activity probability)? 

[M, D] = size(Y); 
[~, N, G] = size(Phis); 
activeMean = 1; 
activeVar = 1; 

%RunOpt = Options('smooth_iters', 100, 'inner_iters', 25, 'verbose', false);
RunOpt = Options('smooth_iters', 5, 'inner_iters', 15, 'verbose', false); %default settings

% Can try with and without this artificial ~temporal correlation
%[~, sortInds] = sort(vecnorm(Y(:,1:D))); 
%Y2 = Y(:,sortInds); 
%for i = 1:D
%yAMP1(i) = {Y2(:,i)};
%phiAMP1(i) = {phi};
%end

for d = 1:D
    yAMP(d) = {Y(:,d)};
    phiAMP(d) = {Phis(:,:,group_indices(d))};
end    

SigGenObj = SigGenParams('M', M, 'N', N, 'T', D, 'version', 'R', 'zeta', ...
    activeMean, 'sigma2', activeVar, 'lambda', kGuess/N);     
Params = ModelParams(SigGenObj, AWGN);
[x_hat, v_hat, lambda_hat] = sp_mmv_wrapper_fxn(yAMP, phiAMP, Params, RunOpt);
X = [x_hat{:}]; 