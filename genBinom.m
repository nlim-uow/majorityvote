function [ c ] = genBinom(a,b)

b=floor(b);
if(a<=(b-1))
    if(a<0)
       c=(-1)^b * exp(gammaln(b-a)-gammaln(-a)-gammaln(b+1)); 
    else 
       disp('Unexpected case, but it theoretically possible (negatively correlated with small p)')
       if(a==floor(a)) 
           c=0;
       else
           ceilA=ceil(a);
           floorA=floor(a);
           c=exp(gammaln(a+1)-gammaln(a-floorA)+gammaln(b-a)-gammaln(ceilA-a)-gammaln(b+1));
       end
    end
else

c=(1/(a+1))/beta(b+1,a-b+1);

end