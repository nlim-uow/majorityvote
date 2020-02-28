function [cummProb] = cummBetaProb(n,p,psi)
%Implementation of the 'Polya distribution CDF that allows for
%negative coefficients
cummProb=0;
for i=ceil((n+1)/2):n
    cummProb=cummProb+exp(gammaln(n+1)-gammaln(i+1)-gammaln(n-i+1)+genAscFactln(p,i,psi)+genAscFactln(1-p,n-i,psi)-genAscFactln(1,n,psi));
end
%assuming that breaking ties randomly has a 1/2 probability of getting it
%right, we consider the cases where ties occur

if (mod(n,2)==0)
    i=n/2;
    cummProb=cummProb+exp(gammaln(n+1)-gammaln(i+1)-gammaln(n-i+1)+genAscFactln(p,i,psi)+genAscFactln(1-p,n-i,psi)-genAscFactln(1,n,psi))/2;
end

