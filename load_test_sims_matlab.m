%Pavan Kota

%Matlab baselines script
expFolderPath = '.\20-11-30_noisetol_redo\'; 
addpath(expFolderPath)

fileBase = {'20-09-08_phiUnif_Var', '_M10_N20_k3_D100_lamTot2_G1.mat'};

paramSweep = {'0.001',  '0.0021544346900318843',  '0.004641588833612777', '0.01', '0.021544346900318832', '0.046415888336127774', '0.1', ...
               '0.21544346900318823',  '0.46415888336127775', '1.0'};
AWGN = 0.01; % Noise variance for additive white gaussian noise
B = 3; % number of baselines
saveStem = 'results_Matlab_withL1SMV_';

for f = 1:length(paramSweep)
    %load(files{f})
    fileCurrent = [fileBase{1}, (paramSweep{f}), fileBase{2}];    
    fileLoad = [expFolderPath, fileCurrent];  
    load(fileLoad)
    [N, T] = size(allLam); 
    D = size(allX, 2); 
    if min(sum(allLam~=0))==max(sum(allLam~=0))
        kGuess = min(sum(allLam~=0)); % fixed sparsity
    else
        error('Sparsity is different between trials')
    end
    
    lamB = zeros(N, T, B); 
    lamCosSim = zeros(T, B); 
    lamRelL2err = zeros(T, B); 
    timeBaselines = zeros(T, B); 
    for t = 1:T
        sumTrue = sum(sum(allX(:,:,t))); 
        rxTrue = sum(max(abs(allX(:,:,t)), [], 2));
        t0 = tic;
        %lamB(:,t,1) = sum(amp_mmv(allY(:,:,t), allPhi(:,:,:,t), allGroupIndices(t,:), AWGN, kGuess), 2)/D;
        %timeBaselines(t,1) = toc(t0); 
        
        %New SMV oracle (Jan. 2021)
        lamB(:,t,1) = cvx_l1oracle_smv(allY(:,:,t), allPhi(:,:,:,t),  allGroupIndices(t,:), sumTrue)/D;
        %lamB(:,t,1) = cvx_l1oracle_smv(allY(:,:,t), allPhi(:,:,:,t),  sumTrue)/D;
        timeBaselines(t,1) = toc(t0);         
        t0 = tic; 
        lamB(:,t,2) = sum(cvx_l1oracle(allY(:,:,t), allPhi(:,:,:,t), allGroupIndices(t,:), sumTrue),2)/D;
        timeBaselines(t,2) = toc(t0); 
        t0 = tic;
        lamB(:,t,3) = sum(cvx_rxoracle(allY(:,:,t), allPhi(:,:,:,t), allGroupIndices(t,:), rxTrue),2)/D;
        timeBaselines(t,3) = toc(t0); 
        
        disp(['Trial ' num2str(t) ' complete'])                
        for b = 1:B
            lamCosSim(t,b) = abs(dot(lamB(:,t,b), allLam(:,t))) / (norm(lamB(:,t,b))*norm(allLam(:,t)));
            lamRelL2err(t,b) = norm(lamB(:,t,b) - allLam(:,t))/norm(allLam(:,t));                
        end
        
    end        
    
    save([expFolderPath, saveStem, fileCurrent], 'expFolderPath', 'allLam', 'lamB','lamCosSim', 'lamRelL2err', 'timeBaselines')
    clearvars -except expFolderPath AWGN B saveStem fileBase paramSweep
end
