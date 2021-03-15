%January 9, 2021
%L1 oracle solution to example

load('mixfig_matlab.mat')
allPhi = double(allPhi); % may be type int
[xrec, status] = cvx_l1oracle(allY, allPhi, allGroupIndices, sum(allX(:)));

[N, D] = size(allX);
lamL1 = sum(xrec,2)/D;

%Measurement error comparison
%Confirms L1 and nonnegativity constraints in cvx_l1oracle are not
%compromising measurement error too much
errTrueX = norm(allY - allPhi * allX, 'fro');
errL1 = norm(allY - allPhi * xrec, 'fro');

if errL1 < errTrueX
    disp('L1-oracle finds low measurement error solution with same L1 as vectorized X*, but is wrong (see Fig in manuscript)') 
end