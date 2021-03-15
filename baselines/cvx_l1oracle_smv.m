%Pavan Kota
%First created: January 14, 2021

%CVX - L1-oracle (SMV) 

function [x, cvx_status] = cvx_l1oracle_smv(Y, Phi, groupInds, sumTrue)
[M, N, G] = size(Phi); 
if G > 1
    warning('Note: G>1 means this algorithm is more like an MMV solver with G=D')
end

%treat as one lumped SMV problem
Ysum = zeros(M, G); 
for g = 1:G  
    Ysum(:,g) = sum(Y(:, groupInds==g), 2); 
end

cvx_begin quiet
    variable x(N,G)   
    minimize (norm(Phi*x-Ysum))        
    x >= 0    
    sum(x(:)) == sumTrue
cvx_end

if G > 1
    x = sum(x,2); % return a single lumped signal
end

end
