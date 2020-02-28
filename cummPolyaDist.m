function [ cummProb ] = cummPolyaDist(n,p,rho)
%Polya Distribution according to K.K Ladha
cummProb=0;
if(rho==0)
    cummProb=cummBinnProb(n,p);
elseif(rho==1)
    cummProb=p;    
else
    delta=rho/(1-rho);
    for i=ceil((n+1)/2):n
        cummProb=cummProb+exp(genBinomln(-p/delta,i)+genBinomln((p-1)/delta,n-i)-genBinomln(-1/delta,n));
    end
    if(mod(n,2)==0)
        i=n/2;
        cummProb=cummProb+exp(genBinomln(-p/delta,i)+genBinomln((p-1)/delta,n-i)-genBinomln(-1/delta,n))/2;
    end

end

