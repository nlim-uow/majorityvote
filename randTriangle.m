for didx=1:15
    d=2^didx;
%    d=3;
    tries=10000;
    results=zeros(1,tries);
    for i=1:tries
      u=1/sqrt(d)*randn(1,d);
      v=1/sqrt(d)*randn(1,d);
      w=1/sqrt(d)*randn(1,d);
      u=u/norm(u);
      v=v/norm(v);
      w=w/norm(w);
      uA=(1-u*(v+w)'+v*w')/(2*sqrt(1-u*(v+w)'+(u*v')*(u*w')));
      vA=(1-v*(u+w)'+u*w')/(2*sqrt(1-v*(u+w)'+(v*u')*(v*w')));
      wA=(1-w*(v+u)'+v*u')/(2*sqrt(1-w*(u+v)'+(w*u')*(w*v')));
      mVec=min([uA vA wA]);
      mAngle=acos(mVec);
      results(i)=mAngle/pi*180;
    end
    figure;
    histfit(results,100,'gev');
end