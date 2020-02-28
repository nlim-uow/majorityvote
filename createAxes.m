%% Create Axes

function [fig, myAxes] = createAxes( layout, varargin )
%UNTITLED9 Summary of this function goes here
sLayout=size(layout);
if (sLayout(1)~=1 | sLayout(2)~=2)
    error 'layoutError, it need to be a 1x2 matrix'
end
axesCount=layout(1)*layout(2);
myAxes=gobjects(1,axesCount);
tempCount=1:axesCount;
block=reshape(tempCount,[layout(2) layout(1)])';
vertLayout=layout(1);
horzLayout=layout(2);
parser=inputParser;


fig=figure;
figUnits='inches';
hSize=9;
vSize=13.5;
myTitle='';
sharedLegend=true;
titleHeight=1;
legendHeight=1;
leftOffset=0;
margins=0.6;
vMargin=1.25;

addOptional(parser,'hSize',hSize);
addOptional(parser,'margins',margins);
addOptional(parser,'leftOffset',leftOffset);
addOptional(parser,'legendHeight',legendHeight);
addOptional(parser,'sharedLegend',true);
addOptional(parser,'titleHeight',titleHeight);
addOptional(parser,'title',myTitle);
addOptional(parser,'vSize',vSize);
addOptional(parser,'figUnits',figUnits);
addOptional(parser,'vMargin',vMargin);

addRequired(parser,'layout');

parse(parser,layout,varargin{:});
hSize=parser.Results.hSize;
vSize=parser.Results.vSize;
figUnits=parser.Results.figUnits;
margins=parser.Results.margins;
leftOffset=parser.Results.leftOffset;
sharedLegend=parser.Results.sharedLegend;
titleHeight=parser.Results.titleHeight;
vMargin=parser.Results.vMargin;
myTitle=parser.Results.title;
layout=parser.Results.layout;


fig.Units=figUnits;
fig.Position=[0.5 0.5 hSize vSize];
plotBot=1.5*legendHeight/vSize;
plotTop=1-titleHeight/vSize;
plotLeft=(margins+leftOffset)/hSize;
plotRight=1-margins/hSize;
plotHSize=(plotRight-plotLeft-(horzLayout-1)*margins/hSize)/horzLayout;
plotVSize=(plotTop-plotBot -(vertLayout-1)*vMargin*margins/vSize)/vertLayout;
for i=1:horzLayout
    axesHPos(block(:,i))=plotLeft+(i-1)*(plotHSize+margins/hSize);
end
for i=1:vertLayout
    axesVPos(block(i,:))=plotTop-i*(plotVSize)- (i-1)*vMargin*margins/vSize;
end

annotation(fig,'textbox',...
    [0.05 0.99-titleHeight/vSize 0.9 titleHeight/vSize],...
    'String',{myTitle},...
    'LineStyle','none',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',13,...
    'FitBoxToText','off');


for i=1:axesCount
    myAxes(i)=axes;
    myAxes(i).Position=[axesHPos(i) axesVPos(i) plotHSize plotVSize];
end

end