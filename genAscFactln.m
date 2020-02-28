function [lnSum] = genAscFactln(p,x,psi)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
outSum=0;
if (p<=0) 
    error('p has to be strictly greater than 0');
end
if (x>0)
    if (p+(x-1)*psi<=0)
        error('generalized factorial has negative terms, check x and psi')
    end
    for i=1:x
        outSum=outSum+log(p+(i-1)*psi);
    end
end
lnSum=outSum;

