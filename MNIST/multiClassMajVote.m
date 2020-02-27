function [ outLabels ] = multiClassMajVote( results, classes)
[obs,ensSize]=size(results);
[noClasses]=size(classes,2);
tempArray=zeros(obs,noClasses);
for i=1:noClasses
    tempArray(:,i)=sum(results==classes(i),2);
end
[temp,Idx]=sort(tempArray,2,'Descend');
ties=(temp(:,1)==temp(:,2));
index=1:obs;
tieIndex=index(ties);
noTies=sum(ties);
coinFlips=randsample(1:2,noTies,'true')';
outLab=Idx(:,1);
for(i=1:noTies)
   outLab(tieIndex(i))=Idx(tieIndex(i),coinFlips(i));   
end
outLabels=zeros(obs,1);
for(i=1:obs)
    outLabels(i)=classes(outLab(i));
end