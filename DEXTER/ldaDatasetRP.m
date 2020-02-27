subspaces=[20 50 100 200];
kCount=size(subspaces,2);
runCount=30;
ensSize=100;

RSSBase=0;
RSSens=zeros(runCount,kCount,ensSize);
RSSens2=zeros(runCount,kCount,ensSize);
accSub=zeros(runCount,kCount,ensSize);
accEns=zeros(runCount,kCount,ensSize);
accEns1=zeros(runCount,kCount,ensSize);
accEns2=zeros(runCount,kCount,ensSize);
accEns3=zeros(runCount,kCount,ensSize);
accEns4=zeros(runCount,kCount,ensSize);
accEns5=zeros(runCount,kCount,ensSize);
valAcc=zeros(runCount,kCount,ensSize);

accBase=0;
r2=zeros(runCount,kCount);
r=zeros(runCount,kCount);
r3=zeros(runCount,kCount);
psi3=zeros(runCount,kCount);
r4=zeros(runCount,kCount);
psi4=zeros(runCount,kCount);
psi=zeros(runCount,kCount);
psi2=zeros(runCount,kCount);
%betaDist=zeros(1,kCount,2);
pk=zeros(runCount,kCount);
U4=0;
UUT2=0;

fileName='dexter_train.data';
        lines = dataread('file', fileName, '%s', 'delimiter', '\n', 'bufsize', 655350);
        obs=length(lines)
        data=zeros(obs,20000);

% for each of the record, obtain the index, and update the data matrix 
        for i=1:obs
            rec=sscanf(char(lines(i)),'%d:%d');
            index=rec([1:2:end]);
            val=rec([2:2:end]);
            data(i,index)=val;
        end
        % Need to remove zero variance indices or it causes problems for the
        % ensemble
        
fileName='dexter_valid.data';
        lines = dataread('file', fileName, '%s', 'delimiter', '\n', 'bufsize', 655350);
        obs=length(lines)
        tData=zeros(obs,20000);

        
% for each of the record, obtain the index, and update the data matrix 
        for i=1:obs
            rec=sscanf(char(lines(i)),'%d:%d');
            index=rec([1:2:end]);
            val=rec([2:2:end]);
            tData(i,index)=val;
        end        
vData=tData;

d=size(data,2);
indices=1:d;
nzVarIndex=indices(var(data,1)>0);
tData=tData(:,nzVarIndex);
data=data(:,nzVarIndex);
vData=vData(:,nzVarIndex);
d=size(data,2);

y=load('dexter_train.labels');
yTest=load('dexter_valid.labels');
yVal=load('dexter_valid.labels');



%UUT2=mean(sum(wRot.^2.*tData.^2,2)*d);
baseW=zeros(1,d);
nTrain=size(data,1);
nTest=size(tData,1);
nVal=size(vData,1);

wBase=LDA(data,y);
lBase=[ones(nTest,1) tData]*wBase';
yBase=lBase(:,1)<lBase(:,2);
yBase=2*yBase-1;
accuracy=sum(yBase==yTest)/nTest;
accBase=accuracy;

baseW=(wBase(1,2:end)-wBase(2,2:end));
baseW=baseW/norm(baseW);

%lIdeal=tData*(wRot)';
%yIdeal=lIdeal>0;
%accIdeal(sNo,thetaNo)=(sum(yIdeal(1:testSize)==0) + sum(yIdeal(testSize+1:testSize*2)==1))/(testSize*2);

idealW=zeros(1,d);
if sum((baseW-idealW).^2)<2
    RSSBase=sum((baseW-idealW).^2);
else
    %% Sometimes n = -n, from the way baseW was created; 
    RSSBase=sum((baseW+idealW).^2);
end



for runNo=1:runCount
    runNo
subWeights=zeros(kCount,2,d);
bias=zeros(kCount,2);
subWeights2=zeros(kCount,2,d);
bias2=zeros(kCount,2);
for kNo=1:kCount
    k=subspaces(kNo)
    subspaceCount=zeros(1,d);
    yEnsRec=zeros(nTest,ensSize);
    yEnsScore=zeros(nTest,2,ensSize);
    indices=randperm(d);
    subTracker=0;
    weightsSub=zeros(ensSize,2,d);
    
    
  for i=1:ensSize
%    if(mod(i,d/k)==0) 
%       indices=randperm(d); 
%       subTracker=0;
%    end
%    subspace=indices(subTracker*k+1:(subTracker+1)*k);
%    subTracker=subTracker+1;
%%Randomized rather than ordered
    valid=0;
    tries=0;
    while valid==0
    rProj=randproj(k,d);
    subspace=randperm(d,k);
    wSub=LDA(data*rProj',y);
    if sum(isnan(wSub))==0
        valid=1;
    else 
        tries=tries+1;
        if tries>10
            valid=1;
            wSub(1,2:end)=randn(1:k);
            wSub(2,2:end)=-wSub(1,2:end);
            wSub(1,1)=0;
            wSub(2,1)=0;
        end
    end
    end
    normWsub=norm(wSub(1,2:end));
    wSub=wSub/normWsub;
    subspaceCount(subspace)=subspaceCount(subspace)+1;
    lValSub=[ones(nVal,1) vData*rProj']*wSub';
    yValSub=lValSub(:,1)<lValSub(:,2);
    yValSub=yValSub*2-1;
    valAcc(kNo,i)=sum(yValSub==yVal)/nVal;
    
    lSub=[ones(nTest,1) tData*rProj']*wSub';
    ySub=lSub(:,1)<lSub(:,2);
    ySub=2*ySub-1;
    accuracy=sum(ySub==yTest)/nTest;
    accSub(runNo,kNo,i)=accuracy; 
        
    subWeights(kNo,1,subspace)=reshape(subWeights(kNo,1,subspace),1,[])+wSub(1,2:end);
    subWeights(kNo,2,subspace)=reshape(subWeights(kNo,2,subspace),1,[])+wSub(2,2:end);
    bias(kNo,1)=bias(kNo,1)+wSub(1,1);
    bias(kNo,2)=bias(kNo,2)+wSub(2,1);  
    weightsSub(i,1,subspace)=wSub(1,2:end);
    weightsSub(i,2,subspace)=wSub(2,2:end);
    

 %   r(sNo,thetaNo,kNo,i)=corr(yEns,ySub);

 
 
    
    if (sum(isnan(lSub(:,1)))+sum(isnan(lSub(:,2))))==0
        yEnsScore(:,1,i)=lSub(:,1);
        yEnsScore(:,2,i)=lSub(:,2);
    end

    
    
    yEnsRec(:,i)=ySub;  
    yEns=2*((sum(yEnsRec(:,1:i),2)/i)>0)-1;
    %break ties randomlly
    tieCount=sum((sum(yEnsRec(:,1:i),2)/i)==0);
    yEns((sum(yEnsRec(:,1:i),2)/i)==0)=(2*randi(2,tieCount,1)-2)-1;
%   
    accuracy=sum(yEns==yTest)/nTest;
    accEns(runNo,kNo,i)=accuracy;       
%    ensW=subWeights(kNo,1,:)/norm(subWeights(kNo,1,:));
%   RSSens(kNo,i)=sum((ensW-idealW).^2);

    % The next line controls how the weights are determined
%    penalty=1/testSize;
    penalty=0.5;
    valAccW=(reshape(valAcc(kNo,1:i),1,i)-penalty);
    valAccTemp=min(reshape(valAcc(kNo,1:i),1,[]),1-1/(nVal));
    valAccW2=log(valAccTemp./(1-valAccTemp));
%    valAccW=log((reshape(valAcc(sNo,thetaNo,kNo,1:i),1,[])-penalty)./(1-((reshape(valAcc(sNo,thetaNo,kNo,1:i),1,[])-penalty))));
%    valAccW=exp(reshape(valAcc(sNo,thetaNo,kNo,1:i),1,[])-1); 
    

    yEns1=(reshape(yEnsScore(:,1,1:i),nTest,i)*(ones(1,i))'/sum(ones(1,i))<reshape(yEnsScore(:,2,1:i),nTest,i)*(ones(1,i))'/sum(ones(1,i)));
    yEns1=2*yEns1-1;
    yEns1tie=(reshape(yEnsScore(:,1,1:i),nTest,i)*(ones(1,i))'/sum(ones(1,i))==reshape(yEnsScore(:,2,1:i),nTest,i)*(ones(1,i))'/sum(ones(1,i)));
    tieCount=sum(yEns1tie);
    yEns1(yEns1tie)=2*randi(2,tieCount,1)-3;
    accuracy=sum(yEns1==yTest)/nTest;
    accEns1(runNo,kNo,i)=accuracy;  
    

    
    yEns2=2*(yEnsRec(:,1:i)*(valAccW2)'/sum(valAccW2)>0)-1;
    yEns2tie=(yEnsRec(:,1:i)*(valAccW2)'/sum(valAccW2)==0);
    tieCount=sum(yEns2tie);
    yEns2(yEns2tie)=2*(randi(2,tieCount,1))-3;
    accuracy=sum(yEns2==yTest)/nTest;
    accEns2(runNo,kNo,i)=accuracy; 
    
    yEns3=2*(yEnsRec(:,1:i)*valAccW'/sum(valAccW)>0)-1;
    yEns3tie=(yEnsRec(:,1:i)*valAccW'/sum(valAccW)==0);
    tieCount=sum(yEns3tie);
    yEns3(yEns3tie)=2*randi(2,tieCount,1)-3;
    accuracy=sum(yEns2==yTest)/nTest;
    accEns3(runNo,kNo,i)=accuracy; 
    
    
    
    yEns4=(reshape(yEnsScore(:,1,1:i),nTest,i)*(valAccW)'/sum(valAccW)<reshape(yEnsScore(:,2,1:i),nTest,i)*(valAccW)'/sum(valAccW));
    yEns4=2*yEns4-1;
    yEns4tie=(reshape(yEnsScore(:,1,1:i),nTest,i)*(valAccW)'/sum(valAccW)==reshape(yEnsScore(:,2,1:i),nTest,i)*(valAccW)'/sum(valAccW));
    tieCount=sum(yEns4tie);
    yEns4(yEns4tie)=2*randi(2,tieCount,1)-3;
    accuracy=sum(yEns4==yTest)/nTest;
    accEns4(runNo,kNo,i)=accuracy; 
    
    yEns5=(reshape(yEnsScore(:,1,1:i),nTest,i)*(valAccW2)'/sum(valAccW2)<reshape(yEnsScore(:,2,1:i),nTest,i)*(valAccW2)'/sum(valAccW2));
    yEns5=2*yEns5-1;
    yEns5tie=(reshape(yEnsScore(:,1,1:i),nTest,i)*(valAccW2)'/sum(valAccW2)==reshape(yEnsScore(:,2,1:i),nTest,i)*(valAccW2)'/sum(valAccW2));
    tieCount=sum(yEns5tie);
    yEns5(yEns5tie)=2*randi(2,tieCount,1)-3;
    accuracy=sum(yEns5==yTest)/nTest;
    accEns5(runNo,kNo,i)=accuracy; 
    
    
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
    
    
    
    
 %   weightedWeights=reshape(weightsSub(1:i,1,:),i,d)'*valAccW';
 %   unweightedWeights=reshape(weightsSub(1:i,1,:),i,d)'*ones(i,1);
 %   enswW=weightedWeights'/norm(weightedWeights);
 %   ensuW=unweightedWeights'/norm(unweightedWeights);
    
%    ensW=subWeights(kNo,:)/norm(subWeights(kNo,:));
%    RSSens(sNo,thetaNo,kNo,i)=sum((ensuW-idealW).^2);
%    RSSens2(sNo,thetaNo,kNo,i)=sum((enswW-idealW).^2);

  end
%     counter=0;
%     polyaData=zeros(100*ensSize,2);
%   for i=1:ensSize
%     index1=randsample(testSize,50);
%     index2=randsample(testSize,50);
%     indice=randsample(ensSize,i);
%     polyaData(counter+1:counter+100,1)=sum([(yEnsRec(index1,indice)==0); (yEnsRec(index2+testSize,indice)==1)],2);
%     polyaData(counter+1:counter+100,2)=i-sum([(yEnsRec(index1,indice)==0); (yEnsRec(index2+testSize,indice)==1)],2);
%     counter=counter+200;
%   end
%   [params]=polya_moment_match(polyaData);
%   betaDist(sNo,thetaNo,kNo,1)=params(1);
%   betaDist(sNo,thetaNo,kNo,2)=params(2);

  
  ensCor=corr(yEnsRec(:,:),yEnsRec(:,:));
  ensCor(isnan(ensCor))=1;
  
  r(runNo,kNo)=(sum(sum(ensCor))-ensSize)/ensSize/(ensSize-1)/2; 
  
  psi(runNo,kNo)=r(runNo,kNo)/(1-r(runNo,kNo));
  
%  ensCor=corr(weightsSub(:,:)',weightsSub(:,:)');
%  ensCor(isnan(ensCor))=1;
  
  r2(runNo,kNo)=k/(2*d-k);
  psi2(kNo)=r2(runNo,kNo)/(1-r2(runNo,kNo));
  
%  temp=reshape(yEnsScore(:,1,:),nTest,[]);
%  ensCor=corr(temp,temp);
%  ensCor(isnan(ensCor))=1;
%  psi3(sNo,thetaNo,kNo)=r3(sNo,thetaNo,kNo)/(1-r3(sNo,thetaNo,kNo));
  
  pk(runNo,kNo)=mean(accSub(runNo,kNo,:));
  
%  ensCor=corr(yEnsScore,yEnsScore);
%  r3(sNo,thetaNo,kNo)=(sum(sum(ensCor))-ensSize)/ensSize/(ensSize-1)/2;
%  psi3(sNo,thetaNo,kNo)=r3(sNo,thetaNo,kNo)/(1-r3(sNo,thetaNo,kNo));
  
%  pk(sNo,thetaNo,kNo)=mean([sum(1-yEnsRec(1:testSize,:),2); sum(yEnsRec(testSize+1:testSize*2,:),2)])/ensSize;
  
rhoTotal=0;
qTotal=0;
count=0;
for i=1:ensSize
    for j=i+1:ensSize
        N11=sum(yEnsRec(yEnsRec(:,j)==yTest,i)==yTest(yEnsRec(:,j)==yTest));
        N01=sum(yEnsRec(yEnsRec(:,j)==yTest,i)~=yTest(yEnsRec(:,j)==yTest));
        N10=sum(yEnsRec(yEnsRec(:,j)~=yTest,i)==yTest(yEnsRec(:,j)~=yTest));
        N00=sum(yEnsRec(yEnsRec(:,j)~=yTest,i)~=yTest(yEnsRec(:,j)~=yTest));
        rho=(N11*N00 - N10*N01)/sqrt((N10+N11)*(N00+N01)*(N11+N01)*(N00+N10));
        qYule=(N11*N00 - N10*N01)/(N11*N00 + N10*N01);
        count=count+1;
        rhoTotal=rhoTotal+rho;
        qTotal=qTotal+qYule;
    end
end
r4(runNo,kNo)=rhoTotal/count;
psi4(runNo,kNo)=r4(runNo,kNo)/(1-r4(runNo,kNo)); 
r3(runNo,kNo)=qTotal/count;
psi3(runNo,kNo)=r3(runNo,kNo)/(1-r3(runNo,kNo)); 



%  if (psi2(kNo)<(pk(kNo)-1)/(ensSize-1))
%      psi2(kNo)=(pk(kNo)-1)/ensSize
%  end
%  if (psi(kNo)<(pk(kNo)-1)/(ensSize-1))
%      psi(kNo)=(pk(kNo)-1)/ensSize
%  end
%  if (psi3(runNo,kNo)<(pk(kNo)-1)/(ensSize-1))
%      psi3(runNo,kNo)=(pk(kNo)-1)/ensSize
%  end
end
end
avgAccEns=reshape(mean(accEns,1),kCount,ensSize);
avgAccEns1=reshape(mean(accEns1,1),kCount,ensSize);
avgAccEns3=reshape(mean(accEns3,1),kCount,ensSize);
avgAccEns4=reshape(mean(accEns4,1),kCount,ensSize);
avgPk=mean(pk,1);
avgR=mean(r,1);
avgR2=mean(r2,1);
avgR3=mean(r3,1);
avgR4=mean(r4,1);


close gcf;
sTitle=sprintf('Ensemble Accuracy vs Ensemble Size for Dataset: Arcene, Training size=%d, Test Size=%d, d=%d',nTrain, nTest,d);
[fig myAxes]=createAxes([2 2],'title',sTitle,'hSize',7.5,'vSize',8.5,'leftOffset',0,'legendHeight',0.75,'vMargin',1.25);
myPlots=gobjects(1,5);
i=0;
    for kNo=1:kCount
        k=subspaces(kNo);
        i=i+1;
        axes(myAxes(i));
        myPlots(1)=plot(1:ensSize,reshape(avgAccEns(kNo,:),[1 ensSize]));
        hold on;
        myPlots(2)=plot(1:ensSize,reshape(avgAccEns1(kNo,:),[1 ensSize]),'-x','MarkerSize',2);
        myPlots(3)=plot(1:ensSize,reshape(avgAccEns3(kNo,:),[1 ensSize]),'-s','MarkerSize',2);
        myPlots(4)=plot(1:ensSize,reshape(avgAccEns4(kNo,:),[1 ensSize]),'-^','MarkerSize',2);
        myPlots(5)=plot(1:ensSize,accBase*ones(1,ensSize),'--');
        xlim([1 ensSize]);
        ylim([0.5 1]);
        ylabel('Ensemble Accuracy');
        xlabel('Ensemble Size');       
        ylim([0.5 1]);
            title(sprintf('Subspace=%d',k));
    end
    legendCell={'Empirical Majority Vote','Empirical (soft vote)','Weighted Majority Vote','Weighted soft vote','Base Classifier Accuracy'};
    legend(myPlots,legendCell,'Location',[0.4 0.045 0.3 0.04]);
    drawnow;
    fileName=sprintf('weightedVote_dexterRP.fig');
    savefig(fileName);
    fileName=sprintf('weightedVote_dexterRP.eps');
    saveas(gcf,fileName,'epsc');


        close gcf;

      
        
        
sTitle=sprintf('Ensemble Accuracy vs Ensemble Size for ARCENE');
[fig myAxes]=createAxes([2 2],'title',sTitle,'hSize',7.5,'vSize',8.5,'leftOffset',0,'legendHeight',0.8,'vMargin',1.25);
myPlots=gobjects(1,6);
plotNo=0;
    for kNo=1:kCount
        k=subspaces(kNo);
        plotNo=plotNo+1;
        axes(myAxes(plotNo));
        myPlots(1)=plot(1:ensSize,avgAccEns(kNo,:),'marker','x','MarkerSize',2);
        hold on;

        a=zeros(1,ensSize);
   if(avgPk(kNo)==1)
       a=ones(1,ensSize);
       myPlots(2)=plot(a,'-.');
       myPlots(3)=plot(a,'-.');
       myPlots(4)=plot(a,'-.');
       myPlots(5)=plot(a,'-.');
       myPlots(6)=plot(a,'-.');    
   else
%    psi2=sqrt(r2(sNo,thetaNo,kNo))/(1-sqrt(r2(sNo,thetaNo,kNo)));   
    rVote=zeros(1,ensSize);
    rWeight=zeros(1,ensSize);
    rScore=zeros(1,ensSize);
    rBinn=zeros(1,ensSize);
    rMLE=zeros(1,ensSize);
    for i=1:ensSize
         rVote(i)=cummPolyaDist(i,avgPk(kNo),avgR(kNo)); 
         rWeight(i)=cummPolyaDist(i,avgPk(kNo),avgR2(kNo)); 
         rScore(i)=cummPolyaDist(i,avgPk(kNo),avgR3(kNo));
         rDiv(i)=cummPolyaDist(i,avgPk(kNo),avgR4(kNo)); 
         rBinn(i)=cummBinnProb(i,avgPk(kNo)); 
 %        rMLE(i)=cummPolyaDist(i,betaDist(sNo,thetaNo,kNo,1)/(betaDist(sNo,thetaNo,kNo,1)+ betaDist(sNo,thetaNo,kNo,2)), 1/2/(1+betaDist(sNo,thetaNo,kNo,1)+ betaDist(sNo,thetaNo,kNo,2)) ); 
    end
        myPlots(2)=plot(rVote,'-.');
        myPlots(3)=plot(rWeight,'-.');
        myPlots(4)=plot(rScore,'-.');
        myPlots(5)=plot(rBinn,'-.');
        myPlots(6)=plot(rDiv,'-.','LineWidth',1.5);
 %       myPlots(6)=plot(rMLE,'-.');
        
   end
        xlim([1 ensSize]);
        ylim([0.5 1]);
        ylabel('Ensemble Accuracy');
        xlabel('Ensemble Size');
        
        ylim([0.5 1]);
        title(sprintf('Subspace=%d',k));
        end
    
    legendCell={'Empirical Majority Vote','Polya Model (Vote Correlation)','Polya Model (Jaccard Similarity)','Polya Model (Yule Diversity)','Binomial Model (Uncorrelated)','Polya Model (Sneath Diversity)'};
    legend(myPlots,legendCell,'Location',[0.4 0.045 0.3 0.04]);
    drawnow;
    fileName=sprintf('polyaModel_dexterRP.fig');
    savefig(fileName);
    fileName=sprintf('polyaModel_dexterRP.eps');
    saveas(gcf,fileName,'epsc');

    
  save dexterRP.mat  
    