%% Setup the number of projection dimensions and number of runs + ensemble member size
subspaces=[20 50 100 200];
kCount=size(subspaces,2);
runCount=100;
ensSize=100;
%% Set Base classifiers 1=LDA-RS, 2=LDA-RP, 3=SVM-RS, 4=SVM-RP, 5=RF
baseClassifier=5;
for baseClassifier=[1 3 5]
prefix='imdb';
if baseClassifier==1
    suffix='lda_rs';
elseif baseClassifier==2
    suffix='lda_rp';
elseif baseClassifier==3
    suffix='svm_rs';
elseif baseClassifier==4
    suffix='svm_rp';
else
    suffix='rf';
end
filespec=sprintf('%s_%s',prefix,suffix);
RSSens=zeros(runCount,kCount,ensSize);
RSSens2=zeros(runCount,kCount,ensSize);
accSub=zeros(runCount,kCount,ensSize);
accEns=zeros(runCount,kCount,ensSize);
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
pk=zeros(runCount,kCount);


%% Load the data
reloadData=1;
if (reloadData==1)
fileName='labeledBow.train';
        lines = dataread('file', fileName, '%s', 'delimiter', '\n', 'bufsize', 655350);
        obs=length(lines)
        data=sparse(obs,100000);
        y=zeros(obs,1);
% for each of the record, obtain the index, and update the data matrix 
        for i=1:obs
            tokens=split(lines(i),' ');
            if(str2double(tokens(1))>5) 
                y(i)=1;
            else
                y(i)=-1;
            end
            line=join(tokens(2:end));
            rec=sscanf(line,'%d:%d');
            index=rec([1:2:end])+1;
            val=rec([2:2:end]);
            data(i,index)=val;
        end

fileName='labeledBow.test';
        lines = dataread('file', fileName, '%s', 'delimiter', '\n', 'bufsize', 655350);
        obs=length(lines)
        dataStore=sparse(obs,100000);
        yStore=zeros(obs,1);
        for i=1:obs
            tokens=split(lines(i),' ');
            if(str2double(tokens(1))>5) 
                yStore(i)=1;
            else
                yStore(i)=-1;
            end
            line=join(tokens(2:end));
            rec=sscanf(line,'%d:%d');
            index=rec([1:2:end])+1;
            val=rec([2:2:end]);
            dataStore(i,index)=val;
        end    
end

%% Remove zero-variance features
d=size(data,2);
%indices=1:d;
%nzVarIndex=indices(var(data,1)>0);
%dataStore=dataStore(:,nzVarIndex);
%data=data(:,nzVarIndex);
%d=size(data,2);


baseW=zeros(1,d);
nTrain=size(data,1);
nStore=size(dataStore,1);



for runNo=1:runCount
    runNo

%% Split the test data into two sets, estimate parameter on the first half, and measure empirical accuracy on the second half       
indices=1:nStore;
valSample=sort([randsample(indices(yStore==1),floor(size(indices(yStore==1),2)/2)) randsample(indices(yStore==-1),floor(size(indices(yStore==-1),2)/2))]');
testSample=setdiff(indices,valSample);
tData=dataStore(testSample,:);
vData=dataStore(valSample,:);
nTest=size(tData,1);
nVal=size(vData,1);
yVal=yStore(valSample);
yTest=yStore(testSample);

    
subWeights=zeros(kCount,2,d);
bias=zeros(kCount,2);
subWeights2=zeros(kCount,2,d);
bias2=zeros(kCount,2);
for kNo=1:kCount
    k=subspaces(kNo);
    subspaceCount=zeros(1,d);
    yEnsRec=zeros(nTest,ensSize);
    yEnsValRec=zeros(nTest,ensSize);
    yEnsScore=zeros(nTest,2,ensSize);
    indices=randperm(d);
    subTracker=0;
    weightsSub=zeros(ensSize,2,d);
    
    
  for i=1:ensSize


    valid=0;
    tries=0;
    subspace=randperm(d,k);

%%Train/Classify validation data/test data using relevant base classifiers
if baseClassifier==1
    wSub=LDA(data(:,subspace),y);
    lValSub=[ones(nVal,1) vData(:,subspace)]*wSub';
    yValSub=lValSub(:,1)<lValSub(:,2);
    yValSub=yValSub*2-1;   
    lSub=[ones(nTest,1) tData(:,subspace)]*wSub';
    ySub=lSub(:,1)<lSub(:,2);
    ySub=2*ySub-1;
elseif baseClassifier==2
    rProj=randproj(k,d);
    wSub=LDA(data*rProj',y);
    lValSub=[ones(nVal,1) vData*rProj']*wSub';
    yValSub=lValSub(:,1)<lValSub(:,2);
    yValSub=yValSub*2-1;
    
    lSub=[ones(nTest,1) tData*rProj']*wSub';
    ySub=lSub(:,1)<lSub(:,2);
    ySub=2*ySub-1;
elseif baseClassifier==3
    svModel=train(y,sparse(data(:,subspace)),'-q -s 5 -B 1');
    yValSub=predict(yVal,sparse(vData(:,subspace)),svModel,'-q');
    ySub=predict(yTest,sparse(tData(:,subspace)),svModel,'-q');
elseif baseClassifier==4
    rProj=randproj(k,d);
    svModel=train(y,sparse(data*rProj'),'-q -s 5 -B 1');
    yValSub=predict(yVal,sparse(vData*rProj'),svModel,'-q');
    ySub=predict(yTest,sparse(tData*rProj'),svModel,'-q');
else 
    sampleIndices=randsample(nTrain,ceil(nTrain/3));
    bootStrapSample=full(data(sampleIndices,subspace));
    cTree=fitctree(bootStrapSample,y(sampleIndices));
    yValSub=cTree.predict(full(vData(:,subspace)));
    ySub=cTree.predict(full(tData(:,subspace)));
end


    valAcc(runNo,kNo,i)=sum(yValSub==yVal)/nVal;

    accuracy=sum(ySub==yTest)/nTest;
    accSub(runNo,kNo,i)=accuracy; 
    
    yEnsRec(:,i)=ySub; 
    yEnsValRec(:,i)=(yValSub==yVal);
    yEns=2*((sum(yEnsRec(:,1:i),2)/i)>0)-1;
    tieCount=sum((sum(yEnsRec(:,1:i),2)/i)==0);
%% Break ties randomly
    yEns((sum(yEnsRec(:,1:i),2)/i)==0)=(2*randi(2,tieCount,1)-2)-1;
    accuracy=sum(yEns==yTest)/nTest;
    accEns(runNo,kNo,i)=accuracy;       


  end

%% Extract the "diversity" measures for PE Model
  
  ensCor=corr(yEnsRec(:,:),yEnsRec(:,:));
  ensCor(isnan(ensCor))=1;
  
  r(runNo,kNo)=(sum(sum(ensCor))-ensSize)/ensSize/(ensSize-1)/2;  
  psi(runNo,kNo)=r(runNo,kNo)/(1-r(runNo,kNo));
  

  r2(runNo,kNo)=k/(2*d-k);
  psi2(kNo)=r2(runNo,kNo)/(1-r2(runNo,kNo)); 
  
  pk(runNo,kNo)=mean(valAcc(runNo,kNo,:));
  

rhoTotal=0;
qTotal=0;
count=0;
for i=1:ensSize
    for j=i+1:ensSize
        N11=sum(yEnsValRec(:,i)==1 & yEnsValRec(:,j)==1);
        N01=sum(yEnsValRec(:,i)==0 & yEnsValRec(:,j)==1);
        N10=sum(yEnsValRec(:,i)==1 & yEnsValRec(:,j)==0);
        N00=sum(yEnsValRec(:,i)==0 & yEnsValRec(:,j)==0);
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




end
end

%% Average the empirical accuracy over N runs 
pct5AccEns=reshape(prctile(accEns,5,1),kCount,ensSize);
pct95AccEns=reshape(prctile(accEns,95,1),kCount,ensSize);
avgAccEns=reshape(mean(accEns,1),kCount,ensSize);
avgPk=mean(pk,1);
avgR=mean(r,1);
avgR2=mean(r2,1);
avgR3=mean(r3,1);
avgR4=mean(r4,1);


close gcf;

      
%% Plot the Empirical Accuracy vs PE-Model                
        
sTitle=sprintf('Ensemble Accuracy vs Ensemble Size for %s',upper(prefix));
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
    rDiv=zeros(1,ensSize);
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
    fileName=sprintf('polyaModel_%s.fig',filespec);
    savefig(fileName);
    fileName=sprintf('polyaModel_%s.eps',filespec);
    saveas(gcf,fileName,'epsc');

    
    pct5AccEns=reshape(prctile(accEns,5,1),kCount,ensSize);
pct95AccEns=reshape(prctile(accEns,95,1),kCount,ensSize);
stdP=std(reshape((permute(accSub,[1 3 2])),runCount*ensSize,4));
avgP=mean(reshape((permute(accSub,[1 3 2])),runCount*ensSize,4));
stdEnsAcc=mean(std(accEns),3);

rssRec=zeros(4,4);
for kNo=1:kCount
    for i=1:ensSize
         rDiv(i)=cummPolyaDist(i,avgPk(kNo),avgR4(kNo)); 
    end
rssRec(:,kNo)=diag(reshape(sqrt(sum((accEns(:,kNo,[25 50 75 100])-rDiv([25 50 75 100])).^2)./runCount),4,4));
end

rssRec;

%% Plot the mean and 95-5% bootstrap empirical accuracy vs PE-Model

sTitle=sprintf('Ensemble Accuracy vs Ensemble Size for %s',upper(prefix));
[fig myAxes]=createAxes([2 2],'title',sTitle,'hSize',7.5,'vSize',8.5,'leftOffset',0,'legendHeight',0.8,'vMargin',1.25);
myPlots=gobjects(1,4);
plotNo=0;
    for kNo=1:kCount
        k=subspaces(kNo);
        plotNo=plotNo+1;
        axes(myAxes(plotNo));
        myPlots(1)=plot(1:ensSize,avgAccEns(kNo,:),'marker','x','MarkerSize',2);
        hold on;
        myPlots(2)=plot(1:ensSize,pct95AccEns(kNo,:),'--','MarkerSize',1);
        myPlots(3)=plot(1:ensSize,pct5AccEns(kNo,:),'--','MarkerSize',1);
     

        a=zeros(1,ensSize);
   if(avgPk(kNo)==1)
       a=ones(1,ensSize);
       myPlots(2)=plot(a,'-.');
       myPlots(3)=plot(a,'-.');
       myPlots(4)=plot(a,'-.');
   else
    rDiv=zeros(1,ensSize);
    for i=1:ensSize
        
        rDiv(i)=cummPolyaDist(i,avgPk(kNo),avgR4(kNo)); 
    end
        myPlots(4)=plot(rDiv,'-.','LineWidth',1.5);
        
   end
        xlim([1 ensSize]);
        ylim([0.5 1]);
        ylabel('Ensemble Accuracy');
        xlabel('Ensemble Size');
        
        ylim([0.5 1]);
        title(sprintf('Subspace=%d',k));
   end
    
    legendCell={'Empirical Majority Vote','Upper 95% Accuracy','Lower 5% Accuracy','Polya Model (Sneath Diversity)'};
    legend(myPlots,legendCell,'Location',[0.4 0.045 0.3 0.04]);
    drawnow;
    fileName=sprintf('ci_%s.fig',filespec);
    savefig(fileName);
    fileName=sprintf('ci_%s.eps',filespec);
    saveas(gcf,fileName,'epsc');

    
    
  save(filespec)
    

