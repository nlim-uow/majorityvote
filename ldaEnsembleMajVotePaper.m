subspaces=[2 10 50];% 10 50];
sparsities=[1 10];% 2 5 10 20];
angles= [80 85 87.5]; %[45 60 75 80 
nSize=[75 1000];
labelNoise=[0];
nCount=size(nSize,2);
sCount=size(sparsities,2);
noiseCount=size(labelNoise,2);
%nouseCount=1;
thetaCount=size(angles,2);
kCount=size(subspaces,2);
ensSize=100;
d=1000;
doFlipBase=0;
testSize=1000;
valSize=testSize;
runCount=100;

for noiseType=1
RSSBase=zeros(runCount,nCount,noiseCount,sCount,thetaCount);
RSSens=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
RSSens2=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
accSub=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
accEns=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
accEns1=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
accEns2=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
accEns3=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
accEns4=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
accEns5=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,ensSize);

valAcc=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,ensSize);

accIdeal=zeros(runCount,nCount,noiseCount,sCount,thetaCount);
accBase=zeros(runCount,nCount,noiseCount,sCount,thetaCount);
r2=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount);
r=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount);
r3=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount);
r4=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount);
psi4=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount);
psi3=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount);
psi=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount);
psi2=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount);
%betaDist=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount,2);
pk=zeros(runCount,nCount,noiseCount,sCount,thetaCount,kCount);

for runNo=1:runCount
for nNo=1:nCount
    nVal=nSize(nNo)
    n0=nVal;
    n1=nVal;

    n=n0+n1;

for noiseNo=1:noiseCount
    noisePer=labelNoise(noiseNo)
for sNo=1:sCount
    sparsity=sparsities(sNo)
    relevantCor=d/sparsity;
    rotation=[orth(randn(relevantCor)) zeros(relevantCor,d-relevantCor);
              zeros(d-relevantCor,relevantCor) eye(d-relevantCor)];
    wIdeal=[1 zeros(1,d-1)];
    wRot=wIdeal*rotation;
%     U4(sNo)=mean(sum(wRot.^4,2)*1000);
for thetaNo=1:thetaCount
    angle=angles(thetaNo)
    s=d/sparsity;
    data0=[cos(pi-angle/180*pi)*ones(n0,1) randnormalized(n0,d-1)*sin(angle/180*pi)];
    data1=[cos(angle/180*pi)*ones(n1,1) randnormalized(n1,d-1)*sin(angle/180*pi)];
 
dataRaw=[data0; data1];
data=dataRaw*rotation;

tData0=[cos(pi-angle/180*pi)*ones(testSize,1) randnormalized(testSize,d-1)*sin(angle/180*pi)];
tData1=[cos(angle/180*pi)*ones(testSize,1) randnormalized(testSize,d-1)*sin(angle/180*pi)];
tData=[tData0; tData1]*rotation;

vData0=[cos(pi-angle/180*pi)*ones(testSize,1) randnormalized(testSize,d-1)*sin(angle/180*pi)];
vData1=[cos(angle/180*pi)*ones(testSize,1) randnormalized(testSize,d-1)*sin(angle/180*pi)];
vData=[vData0; vData1]*rotation;


% UUT2(sNo,thetaNo)=mean(sum(wRot.^2.*tData.^2,2)*d);
yTest=[zeros(testSize,1); ones(testSize,1)];
yVal=[zeros(testSize,1); ones(testSize,1)];
y=[zeros(n0,1); ones(n1,1)];

misLabelCount=ceil(noisePer*nVal);
misLabelVal=ceil(noisePer*testSize);
if (noiseType==1)
    misLableCount=misLabelCount*2;
    misLabelVal=misLabelVal*2;
end
labelFlip0=randperm(nVal,misLabelCount);
labelFlip1=randperm(nVal,misLabelCount)+nVal;
valFlip0=randperm(testSize,misLabelVal);
valFlip1=randperm(testSize,misLabelVal)+testSize;
y(labelFlip0)=1;

if(noiseType==3) 
    yVal(valFlip1)=0;
    yVal(valFlip0)=1;
    y(labelFlip1)=0;
elseif(noiseType==2) 
    y(labelFlip1)=0;
elseif(noiseType==1)
    yVal(valFlip0)=1;   
end



baseW=zeros(1,d);
if doFlipBase==1
    flip=0;
    for i=1:50
        sdata0=[cos(pi-angle/180*pi)*ones(n0,1) randnormalized(n0,d-1)*sin(angle/180*pi)]; 
        sdata1=[cos(angle/180*pi)*ones(n1,1) randnormalized(n1,d-1)*sin(angle/180*pi)];
        sdata=[sdata0; sdata1]*rotation;
    wBase=LDA(sdata,y);
    lBase=tData*(wBase(2,2:end)-wBase(1,2:end))';
    yBase=lBase>0;
    flip=flip+(sum(yBase(1:testSize)==0) + sum(yBase(testSize+1:testSize*2)==1))/(testSize*2);
    end
    accBase(runNo,nNo,noiseNo,sNo,thetaNo)=flip/50

    baseW=(wBase(1,2:end)-wBase(2,2:end));
    baseW=baseW/norm(baseW);

else
    wBase=LDA(data,y);
    lBase=tData*(wBase(2,2:end)-wBase(1,2:end))';
    yBase=lBase>0;
    flip=(sum(yBase(1:testSize)==0) + sum(yBase(testSize+1:testSize*2)==1))/(testSize*2);
    if(flip<0.5)
        flip=1-flip;
    end
    accBase(runNo,nNo,noiseNo,sNo,thetaNo)=flip;
    baseW=(wBase(1,2:end)-wBase(2,2:end));
    baseW=baseW/norm(baseW);
end

lIdeal=tData*(wRot)';
yIdeal=lIdeal>0;
accIdeal(runNo,nNo,noiseNo,sNo,thetaNo)=(sum(yIdeal(1:testSize)==0) + sum(yIdeal(testSize+1:testSize*2)==1))/(testSize*2);

idealW=wRot/norm(wRot);
if sum((baseW-idealW).^2)<2
    RSSBase(runNo,nNo,noiseNo,sNo,thetaNo)=sum((baseW-idealW).^2);
else
    %% Sometimes n = -n, from the way baseW was created; 
    RSSBase(runNo,nNo,noiseNo,sNo,thetaNo)=sum((baseW+idealW).^2);
end

for kNo=1:kCount
    k=subspaces(kNo);
    subspaceCount=zeros(1,d);
    yEnsRec=zeros(testSize*2,ensSize);
    yEnsScore=zeros(testSize*2,ensSize);
    indices=randperm(d);
    subTracker=0;
    weightsSub=zeros(ensSize,d);
    
    
  for i=1:ensSize
    if(mod(i,d/k)==0) 
       indices=randperm(d); 
       subTracker=0;
    end
    subspace=indices(subTracker*k+1:(subTracker+1)*k);
    subTracker=subTracker+1;
%%Randomized rather than ordered
    subspace=randperm(d,k);
    wSub=LDA(data(:,subspace),y);
    subspaceCount(subspace)=subspaceCount(subspace)+1;
    lValSub=vData(:,subspace)*(wSub(1,2:end)-wSub(2,2:end))';
    yValSub=lValSub(:,1)<0;
    valAcc(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=sum(yValSub==yVal)/(testSize*2);
    
    lSub=tData(:,subspace)*(wSub(1,2:end)-wSub(2,2:end))';
    ySub=lSub(:,1)<0;
    accuracy=sum(ySub==yTest)/(testSize*2);
    accSub(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=accuracy; 
    
    
    
    
    subWeights(kNo,subspace)=(subWeights(kNo,subspace)+(wSub(2,2:end)-wSub(1,2:end))/2);
    
    weightsSub(i,subspace)=(wSub(2,2:end)-wSub(1,2:end))/2;
    
    wEns=zeros(2,d+1);
    wEns(1,1)=mean(data0)*subWeights(kNo,:)'*(-0.5)+log(n0/n);
    wEns(2,1)=mean(data1)*(-subWeights(kNo,:))'*(-0.5)+log(n0/n);
    wEns(1,2:end)=subWeights(kNo,:);
    wEns(2,2:end)=-subWeights(kNo,:);
    lEns=tData*(wEns(1,2:end)-wEns(2,2:end))';
    yEns=lEns(:,1)>0;  
 %   r(sNo,thetaNo,kNo,i)=corr(yEns,ySub);
    accuracy=sum(yEns==yTest)/(testSize*2);
    accEns1(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=accuracy;  
    
    yEnsScore(:,i)=lSub;
    yEnsRec(:,i)=ySub;  
    yEns=(sum(yEnsRec(:,1:i),2)/i)>0.5;
    %break ties randomlly
    tieCount=sum((sum(yEnsRec(:,1:i),2)/i)==0.5);
    yEns((sum(yEnsRec(:,1:i),2)/i)==0.5)=(randi(2,tieCount,1)-1);
%   
    accuracy=sum(yEns==yTest)/(testSize*2);
    accEns(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=accuracy;       
    ensW=subWeights(kNo,:)/norm(subWeights(kNo,:));
    RSSens(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=sum((ensW-idealW).^2);

    % The next line controls how the weights are determined
%    penalty=1/testSize;
    penalty=0.5;
    valAccW=(reshape(valAcc(runNo,nNo,noiseNo,sNo,thetaNo,kNo,1:i),1,[])-penalty);
    valAccTemp=min(reshape(valAcc(runNo,nNo,noiseNo,sNo,thetaNo,kNo,1:i),1,[]),1-1/(testSize*2));
    valAccW2=log(valAccTemp./(1-valAccTemp));
%    valAccW=log((reshape(valAcc(sNo,thetaNo,kNo,1:i),1,[])-penalty)./(1-((reshape(valAcc(sNo,thetaNo,kNo,1:i),1,[])-penalty))));
%    valAccW=exp(reshape(valAcc(sNo,thetaNo,kNo,1:i),1,[])-1); 
    
    
    yEns2=(yEnsRec(:,1:i)*(valAccW2)'/sum(valAccW2)>0.5);
    yEns2tie=(yEnsRec(:,1:i)*(valAccW2)'/sum(valAccW2)==0.5);
    tieCount=sum(yEns2tie);
    yEns2(yEns2tie)=(randi(2,tieCount,1)-1);
    accuracy=sum(yEns2==yTest)/(testSize*2);
    accEns2(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=accuracy; 
    
    yEns3=(yEnsRec(:,1:i)*valAccW'/sum(valAccW)>0.5);
    yEns3tie=(yEnsRec(:,1:i)*valAccW'/sum(valAccW)==0.5);
    tieCount=sum(yEns3tie);
    yEns3(yEns3tie)=(randi(2,tieCount,1)-1);
    accuracy=sum(yEns3==yTest)/(testSize*2);
    accEns3(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=accuracy; 
    
    
    
    yEns4=(yEnsScore(:,1:i)*(valAccW)'/sum(valAccW)<0);
    yEns4tie=(yEnsScore(:,1:i)*(valAccW)'/sum(valAccW)==0);
    tieCount=sum(yEns4tie);
    yEns4(yEns4tie)=(randi(2,tieCount,1)-1);
    accuracy=sum(yEns4==yTest)/(testSize*2);
    accEns4(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=accuracy; 
    
    yEns5=(yEnsScore(:,1:i)*(valAccW2)'/sum(valAccW2)<0);
    yEns5tie=(yEnsScore(:,1:i)*(valAccW2)'/sum(valAccW2)==0);
    tieCount=sum(yEns5tie);
    yEns5(yEns5tie)=(randi(2,tieCount,1)-1);
    accuracy=sum(yEns5==yTest)/(testSize*2);
    accEns5(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=accuracy; 
    
    
    %%Temporary
%    yEns4=(yEnsRec(:,1:i)*valAccW2'/sum(valAccW2)>0.5);
%    yEns4tie=(yEnsRec(:,1:i)*valAccW2'/sum(valAccW2)==0.5);
%    tieCount=sum(yEns4tie);
%    yEns4(yEns4tie)=(randi(2,tieCount,1)-1);
%    accuracy=(sum(yEns4(1:testSize)==0) + sum(yEns4(testSize+1:testSize*2)==1))/(testSize*2);
%    accEns4(sNo,thetaNo,kNo,i)=accuracy; 
%    yEns3=(yEnsScore(:,1:i)*(valAccW)'/sum(valAccW)<0);
%    yEns3tie=(yEnsScore(:,1:i)*(valAccW)'/sum(valAccW)==0);
%    tieCount=sum(yEns3tie);
%    yEns3(yEns3tie)=(randi(2,tieCount,1)-1);
%    accuracy=(sum(yEns3(1:testSize)==0) + sum(yEns3(testSize+1:testSize*2)==1))/(testSize*2);
%    accEns3(sNo,thetaNo,kNo,i)=accuracy; 
    
    
    
    
    weightedWeights=weightsSub(1:i,:)'*valAccW';
    unweightedWeights=weightsSub(1:i,:)'*ones(i,1);
    enswW=weightedWeights'/norm(weightedWeights);
    ensuW=unweightedWeights'/norm(unweightedWeights);
    
%    ensW=subWeights(kNo,:)/norm(subWeights(kNo,:));
    RSSens(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=sum((ensuW-idealW).^2);
    RSSens2(runNo,nNo,noiseNo,sNo,thetaNo,kNo,i)=sum((enswW-idealW).^2);

  end
%     counter=0;
%     polyaData=zeros(200*ensSize,2);
%   for i=1:ensSize
%     index1=randsample(testSize,100);
%     index2=randsample(testSize,100);
%     indice=randsample(ensSize,i);
%     polyaData(counter+1:counter+200,1)=sum([(yEnsRec(index1,indice)==0); (yEnsRec(index2+testSize,indice)==1)],2);
%     polyaData(counter+1:counter+200,2)=i-sum([(yEnsRec(index1,indice)==0); (yEnsRec(index2+testSize,indice)==1)],2);
%     counter=counter+200;
%   end
%  [params]=polya_moment_match(polyaData);
%  betaDist(nNo,noiseNo,sNo,thetaNo,kNo,1)=params(1);
%  betaDist(nNo,noiseNo,sNo,thetaNo,kNo,2)=params(2);

  temp=yEnsRec*2-1;
  ensCor=(sum(sum((temp'*temp)/2000))-250)/250/(250-1);
 % ensCor=corr(yEnsRec(:,:),yEnsRec(:,:));
 % r(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=(sum(sum(ensCor))-ensSize)/ensSize/(ensSize-1)/2;  
 r(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=ensCor;
  psi(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=r(runNo,nNo,noiseNo,sNo,thetaNo,kNo)/(1-r(runNo,nNo,noiseNo,sNo,thetaNo,kNo));
  %ensCor=corr(weightsSub(:,:)',weightsSub(:,:)');
  r2(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=k/(2*d-k);
  psi2(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=r2(runNo,nNo,noiseNo,sNo,thetaNo,kNo)/(1-r2(runNo,nNo,noiseNo,sNo,thetaNo,kNo));
  
%  ensCor=corr(yEnsScore,yEnsScore);
%  r3(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=(sum(sum(ensCor))-ensSize)/ensSize/(ensSize-1)/2;
%  psi3(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=r3(runNo,nNo,noiseNo,sNo,thetaNo,kNo)/(1-r3(runNo,nNo,noiseNo,sNo,thetaNo,kNo));
  
  pk(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=mean([sum(1-yEnsRec(1:testSize,:),2); sum(yEnsRec(testSize+1:testSize*2,:),2)])/ensSize;
  
  if (psi2(runNo,nNo,noiseNo,sNo,thetaNo,kNo)<(pk(runNo,nNo,noiseNo,sNo,thetaNo,kNo)-1)/(ensSize-1))
      psi2(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=(pk(runNo,nNo,noiseNo,sNo,thetaNo,kNo)-1)/ensSize;
  end
  if (psi(runNo,nNo,noiseNo,sNo,thetaNo,kNo)<(pk(runNo,nNo,noiseNo,sNo,thetaNo,kNo)-1)/(ensSize-1))
      psi(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=(pk(runNo,nNo,noiseNo,sNo,thetaNo,kNo)-1)/ensSize;
  end
  if (psi3(runNo,nNo,noiseNo,sNo,thetaNo,kNo)<(pk(runNo,nNo,noiseNo,sNo,thetaNo,kNo)-1)/(ensSize-1))
      psi3(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=(pk(runNo,nNo,noiseNo,sNo,thetaNo,kNo)-1)/ensSize;
  end
  rhoTotal=0;
  yuleTotal=0;
  count=0;
  yTest=[zeros(testSize,1);ones(testSize,1)];
  for i=1:ensSize
    for j=i+1:ensSize
        N11=sum(yEnsRec(yEnsRec(:,j)==yTest,i)==yTest(yEnsRec(:,j)==yTest));
        N01=sum(yEnsRec(yEnsRec(:,j)==yTest,i)~=yTest(yEnsRec(:,j)==yTest));
        N10=sum(yEnsRec(yEnsRec(:,j)~=yTest,i)==yTest(yEnsRec(:,j)~=yTest));
        N00=sum(yEnsRec(yEnsRec(:,j)~=yTest,i)~=yTest(yEnsRec(:,j)~=yTest));
        rho=(N11*N00 - N10*N01)/sqrt((N10+N11)*(N00+N01)*(N11+N01)*(N00+N10));
        yule=(N11*N00 - N10*N01)/(N11*N00 + N10*N01);
        count=count+1;
        rhoTotal=rhoTotal+rho;
        yuleTotal=yuleTotal+yule;
    end
  end
  r4(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=rhoTotal/count;
  psi4(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=r4(runNo,nNo,noiseNo,sNo,thetaNo,kNo)/(1-r4(runNo,nNo,noiseNo,sNo,thetaNo,kNo)); 
  
  r3(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=yuleTotal/count;
  psi3(runNo,nNo,noiseNo,sNo,thetaNo,kNo)=r3(runNo,nNo,noiseNo,sNo,thetaNo,kNo)/(1-r3(runNo,nNo,noiseNo,sNo,thetaNo,kNo)); 

  
  
  
end


end

    
    

  

end


end
end

end


  avgAccEns=reshape(mean(accEns,1),nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
  avgAccEns1=reshape(mean(accEns1,1),nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
  avgAccEns3=reshape(mean(accEns3,1),nCount,noiseCount,sCount,thetaCount,kCount,ensSize);
  avgAccEns4=reshape(mean(accEns4,1),nCount,noiseCount,sCount,thetaCount,kCount,ensSize);

  avgPk=reshape(mean(pk,1),nCount,noiseCount,sCount,thetaCount,kCount);
  avgR4=reshape(mean(r4,1),nCount,noiseCount,sCount,thetaCount,kCount);
  avgR3=reshape(mean(r3,1),nCount,noiseCount,sCount,thetaCount,kCount);
  avgR2=reshape(mean(r2,1),nCount,noiseCount,sCount,thetaCount,kCount);
  avgR=reshape(mean(r,1),nCount,noiseCount,sCount,thetaCount,kCount);
  
  avgAccBase=reshape(mean(accBase,1),nCount,noiseCount,sCount,thetaCount);
for nNo=1:nCount
  for noiseNo=1:noiseCount
      for sNo=1:sCount
        for thetaNo=1:thetaCount
            for kNo=1:kCount  
                for i=1:ensSize
                    modelVote(nNo,noiseNo,sNo,thetaNo,kNo,i)=cummPolyaDist(i,avgPk(nNo,noiseNo,sNo,thetaNo,kNo),avgR(nNo,noiseNo,sNo,thetaNo,kNo));
                    modelJacc(nNo,noiseNo,sNo,thetaNo,kNo,i)=cummPolyaDist(i,avgPk(nNo,noiseNo,sNo,thetaNo,kNo),avgR2(nNo,noiseNo,sNo,thetaNo,kNo));
                    modelScore(nNo,noiseNo,sNo,thetaNo,kNo,i)=cummPolyaDist(i,avgPk(nNo,noiseNo,sNo,thetaNo,kNo),avgR3(nNo,noiseNo,sNo,thetaNo,kNo));
                    modelBinn(nNo,noiseNo,sNo,thetaNo,kNo,i)=cummBinnProb(i,avgPk(nNo,noiseNo,sNo,thetaNo,kNo));
                    modelDiv(nNo,noiseNo,sNo,thetaNo,kNo,i)=cummPolyaDist(i,avgPk(nNo,noiseNo,sNo,thetaNo,kNo),avgR4(nNo,noiseNo,sNo,thetaNo,kNo));  
                end
            end
        end
      end
  end
end


alpha=avgPk.*((1-avgR4)./avgR4);
bbeta=(1-avgPk).*((1-avgR4)./avgR4);
betaCDF2=betacdf(0.5,bbeta,alpha);

llabels=cell(5,1);
slabels=cell(5,1);
llabels{1}='Training Size';
llabels{2}='Mislabel Prop';
llabels{3}='Feature Noise';
llabels{4}='\theta';
llabels{5}='Subspace Count';

slabels{1}='n';
slabels{2}='ml';
slabels{3}='s';
slabels{4}='theta';
slabels{5}='k';

paramValues=zeros(5,3);
paramValues(1,:)=nSize*2;
paramValues(2,:)=labelNoise;
paramValues(3,:)=sparsities;
paramValues(4,:)=angles;
paramValues(5,:)=subspaces;

for paramX=5:-1:2
    for paramY=(paramX-1):-1:1
        paramLX=llabels{paramX};
        paramSX=slabels{paramX};
        paramLY=llabels{paramY};
        paramSY=slabels{paramY};
        
        tempParam=1:5;
        tempParam=tempParam(tempParam~=paramX & tempParam~=paramY);
        paramS1=slabels{tempParam(1)};
        paramL1=llabels{tempParam(1)};          
        paramS2=slabels{tempParam(2)};
        paramL2=llabels{tempParam(2)};          
        paramS3=slabels{tempParam(3)};
        paramL3=llabels{tempParam(3)};          
        for param1=1:3
            for param2=1:3
                for param3=1:3
                    close gcf;
                    myPlots=gobjects(1,5);
                    sTitle=sprintf('Ensemble Accuracy vs Ensemble Size for %s=%0.5g, %s=%0.5g, %s=%0.5g',paramL1,paramValues(tempParam(1),param1),paramL2,paramValues(tempParam(2),param2),paramL3,paramValues(tempParam(3),param3));
                    [fig myAxes]=createAxes([3 3],'title',sTitle,'hSize',7.5,'vSize',8.5,'leftOffset',0.5,'legendHeight',0.75,'vMargin',1);
                    plotNo=0;
                    for yPlot=1:3
                        for xPlot=1:3
                        plotNo=plotNo+1;
                        axes(myAxes(plotNo));
                        tempAccEns=permute(avgAccEns,[paramX paramY tempParam 6]);
                        tempAccEns1=permute(avgAccEns1,[paramX paramY tempParam 6]);
                        tempAccEns3=permute(avgAccEns3,[paramX paramY tempParam 6]);
                        tempAccEns4=permute(avgAccEns4,[paramX paramY tempParam 6]);
                        myPlots(1)=plot(1:ensSize,reshape(tempAccEns(xPlot,yPlot,param1,param2,param3,:),[1 ensSize]));
                        hold on;
                        myPlots(2)=plot(1:ensSize,reshape(tempAccEns1(xPlot,yPlot,param1,param2,param3,:),[1 ensSize]),'-x','MarkerSize',2);
                        myPlots(3)=plot(1:ensSize,reshape(tempAccEns3(xPlot,yPlot,param1,param2,param3,:),[1 ensSize]),'-s','MarkerSize',2);
                        myPlots(4)=plot(1:ensSize,reshape(tempAccEns4(xPlot,yPlot,param1,param2,param3,:),[1 ensSize]),'-^','MarkerSize',2);
                        if(paramX==5) 
                            tempAccBase=permute(avgAccBase,[paramY tempParam]);
                            myPlots(5)=plot(1:ensSize,tempAccBase(yPlot,param1,param2,param3)*ones(1,ensSize),'--');
                        else
                            tempParam2=1:4;
                            tempParam2=tempParam2(tempParam2~=paramX & tempParam2~=paramY);
                            tempAccBase=permute(avgAccBase,[paramX paramY tempParam2]);
                            myPlots(5)=plot(1:ensSize,tempAccBase(xPlot,yPlot,param1,param2)*ones(1,ensSize),'--');
                        end               
                        xlim([1 ensSize]);
                        ylim([0.5 1]);
                        ylabel('Ensemble Accuracy');
                        xlabel('Ensemble Size');
                        if xPlot==1
                            text(-0.5*ensSize,0.75,sprintf('%s=%0.5g',paramLY,paramValues(paramY,yPlot)),'Rotation',90,'FontWeight','bold','FontSize',12,'HorizontalAlignment','center')
                        end
                        ylim([0.5 1]);
                        if yPlot==1
                            title(sprintf('%s=%0.5g',paramLX,paramValues(paramX,xPlot)));
                        end
                        end
                    end
                    legendCell={'Empirical Majority Vote','Empirical (soft vote)','Weighted Majority Vote','Weighted soft vote','Base Classifier Accuracy'};
                    legend(myPlots,legendCell,'Location',[0.4 0.045 0.3 0.04]);
                    drawnow;
                    fileName=sprintf('weightedVote_%s_%.5g_%s_%.5g_%s_%.5g_nType_%d.fig',paramS1,paramValues(tempParam(1),param1),paramS2,paramValues(tempParam(2),param2),paramS3,paramValues(tempParam(3),param3),noiseType);
                    savefig(fileName);
                    fileName=sprintf('weightedVote_%s_%.5g_%s_%.5g_%s_%.5g_nType_%d.eps',paramS1,paramValues(tempParam(1),param1),paramS2,paramValues(tempParam(2),param2),paramS3,paramValues(tempParam(3),param3),noiseType);
                    saveas(gcf,fileName,'epsc');
                    close gcf;
                    sTitle=sprintf('Modelled Ensemble Accuracy vs Ensemble Size for %s=%0.5g, %s=%0.5g, %s=%0.5g',paramL1,paramValues(tempParam(1),param1),paramL2,paramValues(tempParam(2),param2),paramL3,paramValues(tempParam(3),param3));
                    [fig myAxes]=createAxes([3 3],'title',sTitle,'hSize',7.5,'vSize',8.5,'leftOffset',0.5,'legendHeight',0.85,'vMargin',1);
                    myPlots=gobjects(1,7);
                    plotNo=0;
                    for yPlot=1:3
                        for xPlot=1:3
                            plotNo=plotNo+1;
                            axes(myAxes(plotNo));
                            tempAccEns=permute(avgAccEns,[paramX paramY tempParam 6]);                            
                            myPlots(1)=plot(1:ensSize,reshape(tempAccEns(xPlot,yPlot,param1,param2,param3,:),[1 ensSize]));
                            hold on;
                            a=zeros(1,ensSize);
                            tempPk=permute(avgPk,[paramX paramY tempParam]);
                            pkVal=tempPk(xPlot,yPlot,param1,param2,param3);
                            if pkVal==1
                                a=ones(1,ensSize);
                                myPlots(2)=plot(a,'-.');
                                myPlots(3)=plot(a,'-.');
                                myPlots(4)=plot(a,'-.');
                                myPlots(5)=plot(a,'-.');
                                myPlots(6)=plot(a,'-.');
                                myPlots(7)=plot(a,'-.');
                            else
                            %    psi2=sqrt(r2(sNo,thetaNo,kNo))/(1-sqrt(r2(sNo,thetaNo,kNo)));   
                            tempAsymptAcc=permute(betaCDF2,[paramX paramY tempParam]);
                            tempModelVote=permute(modelVote,[paramX paramY tempParam 6]);    
                            tempModelJacc=permute(modelJacc,[paramX paramY tempParam 6]);    
                            tempModelScore=permute(modelScore,[paramX paramY tempParam 6]);    
                            tempModelBinn=permute(modelBinn,[paramX paramY tempParam 6]);    
                            tempModelDiv=permute(modelDiv,[paramX paramY tempParam 6]);    

                            myPlots(4)=plot(1:ensSize,reshape(tempModelVote(xPlot,yPlot,param1,param2,param3,:),[1 ensSize]),'-.');
                            myPlots(5)=plot(1:ensSize,reshape(tempModelJacc(xPlot,yPlot,param1,param2,param3,:),[1 ensSize]),'-.');
                            myPlots(3)=plot(1:ensSize,reshape(tempModelScore(xPlot,yPlot,param1,param2,param3,:),[1 ensSize]),'-.');
                            myPlots(2)=plot(1:ensSize,reshape(tempModelDiv(xPlot,yPlot,param1,param2,param3,:),[1 ensSize]),'-.','LineWidth',2);
                            myPlots(6)=plot(1:ensSize,reshape(tempModelBinn(xPlot,yPlot,param1,param2,param3,:),[1 ensSize]),'-.');
                            myPlots(7)=plot(1:ensSize,tempAsymptAcc(xPlot,yPlot,param1,param2,param3)*ones(1,ensSize),'--');
                            end
                            xlim([1 ensSize]);
                            ylim([0.5 1]);
                            ylabel('Ensemble Accuracy');
                            xlabel('Ensemble Size');
                            if xPlot==1
                                text(-0.5*ensSize,0.75,sprintf('%s=%0.5g',paramLY,paramValues(paramY,yPlot)),'Rotation',90,'FontWeight','bold','FontSize',12,'HorizontalAlignment','center')
                            end
                            ylim([0.5 1]);
                            if yPlot==1
                                title(sprintf('%s=%0.5g',paramLX,paramValues(paramX,xPlot)));
                            end
                        end
                    end
                    legendCell={'Empirical Majority Vote','Polya Model (Sneath Diversity)','Polya Model (Yule Diversity)','Polya Model (Vote)','Polya Model (Jaccard Similarity)','Binomial Model (Uncorrelated)','Asymptoptic Accuracy'};%, 'Polya Model (MLE Estimates)'};
                    legend(myPlots,legendCell,'Location',[0.4 0.045 0.3 0.04]);
                    drawnow;
                    fileName=sprintf('polyaModel_%s_%.5g_%s_%.5g_%s_%.5g_nType_%d.fig',paramS1,paramValues(tempParam(1),param1),paramS2,paramValues(tempParam(2),param2),paramS3,paramValues(tempParam(3),param3),noiseType);
                    savefig(fileName);
                    fileName=sprintf('polyaModel_%s_%.5g_%s_%.5g_%s_%.5g_nType_%d.eps',paramS1,paramValues(tempParam(1),param1),paramS2,paramValues(tempParam(2),param2),paramS3,paramValues(tempParam(3),param3),noiseType);
                    saveas(gcf,fileName,'epsc');                   
                end
            end
        end
    end
end   
                    
                    


        
        
        






fileName=sprintf('flipEnsemble_lda_noiseType_%d.mat',noiseType);
save(fileName)
end