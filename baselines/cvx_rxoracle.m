%Pavan Kota
%First created: August 23, 2020

%CVX - rx-oracle 


function [X, cvx_status] = cvx_rxoracle(Y, Phis, group_indices, rxTrue)
D = size(Y, 2);
[M, N, G] = size(Phis); 

cvx_begin quiet
    variable X(N,D)   
    %minimize( norm(phi*X-Y, 'fro') )
    minimize( group_fro(Y, X, Phis, group_indices) ); 
    X >= 0        
    rxnorm(X) <= rxTrue
cvx_end

end


function val = rxnorm(X)
    val = sum(max(abs(X), [], 2)); 
end
