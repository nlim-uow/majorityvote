pct5AccEns=reshape(prctile(accEns,5,1),kCount,ensSize);
pct95AccEns=reshape(prctile(accEns,95,1),kCount,ensSize);
stdP=std(reshape((permute(accSub,[1 3 2])),runCount*ensSize,4))
avgP=mean(reshape((permute(accSub,[1 3 2])),runCount*ensSize,4))
avgR4
stdEnsAcc=mean(std(accEns),3)

rssRec=zeros(4,4);
for kNo=1:kCount
    for i=1:ensSize
         rDiv(i)=cummPolyaDist(i,avgPk(kNo),avgR4(kNo)); 
    end
rssRec(:,kNo)=diag(reshape(sqrt(sum((accEns(:,kNo,[25 50 75 100])-rDiv([25 50 75 100])).^2)./runCount),4,4));
end

rssRec



sTitle=sprintf('Ensemble Accuracy vs Ensemble Size for IMDB');
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
%    psi2=sqrt(r2(sNo,thetaNo,kNo))/(1-sqrt(r2(sNo,thetaNo,kNo)));   
    rDiv=zeros(1,ensSize);
    for i=1:ensSize
        
        rDiv(i)=cummPolyaDist(i,avgPk(kNo),avgR4(kNo)); 
    end
        myPlots(4)=plot(rDiv,'-.','LineWidth',1.5);
 %       myPlots(6)=plot(rMLE,'-.');
        
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
    fileName=sprintf('ci_IMDB_rf.fig');
    savefig(fileName);
    fileName=sprintf('ci_IMDB_rf.eps');
    saveas(gcf,fileName,'epsc');
