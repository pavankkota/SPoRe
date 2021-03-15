%Pavan Kota
%First created: August 23, 2020

%CVX - L1-oracle 


function [X, cvx_status] = cvx_l1oracle(Y, Phis, group_indices, sumTrue)
D = size(Y, 2);
[M, N, G] = size(Phis); 

cvx_begin quiet
    variable X(N,D)   
    %minimize( norm(Phis*X-Y, 'fro') )
    minimize( group_fro(Y, X, Phis, group_indices) ); 
    X >= 0    
    sum(X(:)) == sumTrue % Note - can't do an l1 constraint w
cvx_end

end
