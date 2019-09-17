function [edgeMap,nodeMap,Xnode,Xedge,edgeStruct,y,classP,classR,cordsClass,class ] = createSuperCRF(pathToTrainingImage,pathToSuperpixelImage5x, pathToSuperpixelImageSS1)
%createSuperCRF Create a Conditional Random Field for Histopathology cell
%data
% Input parameters:
%   pathToTrainingImage=cell coordinates of single cell, a table with
%   columns: class = cell classes
%   ('l'=lymphocyte,'c'=cancer,'e'=epidermis,'o'=fibroblast),
%   xMapped5x,yMapped5x = cell x,y coordinates mapped to 5x magnification,
%   xMappedSS1,yMappedSS1 = cell x,y coordinates mapped to 1.25x
%   magnification
%   pathToSuperpixelImage5x= Path to the output of the superpixel classifier (5x
%   magnification)
%   pathToSuperpixelImage1x= Path to  the output of the superpixel classifier (5x
%   magnification)
% Output parameters: Structure of the CRF: edgeMap,nodeMap,Xnode,Xedge,edgeStruct
cellFeaturesAll = readtable(pathToTrainingImage);
indNoLym=~strcmp(cellFeaturesAll.class,'l');
cellFeatures=cellFeaturesAll(~strcmp(cellFeaturesAll.class,'l'),:);%No Lyms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathToSuperpixelImages5x=load(pathToSuperpixelImage5x);
superpixelImage5x=pathToSuperpixelImages5x.L2;
%
pathToSuperpixelImagesSS1=load(pathToSuperpixelImage1x);
superpixelImageSS1=pathToSuperpixelImagesSS1.L2;
%

cordsClass=[cellFeatures.xMapped5x,cellFeatures.yMapped5x];
%
%figure
%imshow(superpixelImage5x)
%hold on
%plot(cellFeatures.xMapped5x,cellFeatures.yMapped5x,'.','MarkerSize',10)

linearInd5x = sub2ind(size(superpixelImage5x), cellFeatures.yMapped5x, cellFeatures.xMapped5x);
superpixelClass5x=superpixelImage5x(linearInd5x);
superpixelClassConverted5x=zeros(1,numel(superpixelClass5x));
%
linearIndSS1 = sub2ind(size(superpixelImageSS1), cellFeatures.yMappedSS1, cellFeatures.xMappedSS1);
superpixelClassSS1=superpixelImageSS1(linearIndSS1);
superpixelClassConvertedSS1=zeros(1,numel(superpixelClassSS1));
%Superpixel
%1 - tumour, 2 - stroma, 3 - epidermis, 4 - lymphocyte cluster
%Single Cell
%1 - tumour, 2 - stroma, 3 - lymphocyte, 4 - epidermis
superpixelClassConverted5x(superpixelClass5x==0)=5;
superpixelClassConverted5x(superpixelClass5x==1)=1;
superpixelClassConverted5x(superpixelClass5x==2)=2;
superpixelClassConverted5x(superpixelClass5x==3)=4;
superpixelClassConverted5x(superpixelClass5x==4)=3;
superpixelClassConverted5x(superpixelClass5x==5)=5;
%
superpixelClassConvertedSS1(superpixelClassSS1==1)=1;
superpixelClassConvertedSS1(superpixelClassSS1==2)=2;
superpixelClassConvertedSS1(superpixelClassSS1==4)=5;
superpixelClassConvertedSS1(superpixelClassSS1==3)=4;
%
numClassesCell=5;
%
classToPredict=cell2mat(cellFeatures.classPredicted);
nNodes=size(classToPredict,1);
endLayer1=nNodes+nNodes;
endLayer2=endLayer1+nNodes;
%
classP=zeros(nNodes+nNodes+nNodes,1);
classP(strcmp(classToPredict,"c"))=1;
classP(strcmp(classToPredict,"o"))=2;
classP(strcmp(classToPredict,"l"))=3;
classP(strcmp(classToPredict,"e"))=4;
%
classP((nNodes+1):(endLayer1))=superpixelClassConverted5x;
classP((endLayer1+1):(endLayer2))=superpixelClassConvertedSS1;
%
class=cell2mat(cellFeatures.class);
classR=zeros(nNodes+nNodes+nNodes,1);
classR(strcmp(class,"c"))=1;
classR(strcmp(class,"o"))=2;
classR(strcmp(class,"l"))=3;
classR(strcmp(class,"e"))=4;
classR((nNodes+1):(endLayer1))=superpixelClassConverted5x;
classR((endLayer1+1):(endLayer2))=superpixelClassConvertedSS1;
nNodesAll=size(classR,1);
%%%
y = reshape(classR,1,size(classR,1),1);
%%%
D = pdist(cordsClass);
distMat= squareform(D);
%

distMat(distMat>4)=0;
distMat(distMat>0)=1;
neighborMat=distMat - diag(diag(distMat));
adj = sparse(neighborMat);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
newNodesSuperpixel=nNodes+nNodes;
edgeSuperpixel=max(adj);
nodeSuperpixel=zeros(newNodesSuperpixel,newNodesSuperpixel);
nodeSuperpixel(1:nNodes,1:nNodes)=neighborMat;
diagNS=diag(nNodes);
nodeSuperpixel(1:nNodes,(nNodes+1):endLayer1)=diag(ones(nNodes,1));
nodeSuperpixel(1:nNodes,(endLayer1+1):endLayer2)=diag(ones(nNodes,1));
nodeSuperpixel((nNodes+1):endLayer1,1:nNodes)=diag(ones(nNodes,1));
nodeSuperpixel((endLayer1+1):endLayer2,1:nNodes)=diag(ones(nNodes,1));
%%%
%nodeSuperpixel(1:nNodes,(nNodes+1):newNodesSuperpixel)=diag(ones(nNodes,1));
%nodeSuperpixel((nNodes+1):newNodesSuperpixel,1:nNodes)=diag(ones(nNodes,1));
%%%
adjAll = sparse(nodeSuperpixel);
nNodesAll=nNodes+nNodes+nNodes;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add bias and Standardize Columns
nInstances=1;
nNodes=size(cordsClass,1);
tied = 1;
%
numClassesCell=5;
numStatesAll=numClassesCell;
%
nodeF = zeros(1,numStatesAll);
nodeF(1,1,classP==1)=1;
nodeF(1,2,classP==2)=1;
nodeF(1,3,classP==3)=1;
nodeF(1,4,classP==4)=1;
nodeF(1,5,classP==5)=1;
%
%
%X = reshape(nodeF,1,1,nNodes);
Xnode = [ones(nInstances,1,nNodesAll)  nodeF];
%Xnode = [ones(nInstances,1,nNodes)];
nNodeFeatures = size(Xnode,2);

% Make nodeMap


nodeMap = zeros(nNodesAll,numStatesAll,nNodeFeatures,'int32');
nodeMap(:,1:numStatesAll,1) = 1;
nodeMap(classP==1,1,2)=2;
nodeMap(classP==2,2,3)=3;
nodeMap(classP==3,3,4)=4;
nodeMap(classP==4,4,5)=5;
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

edgeStruct = UGM_makeEdgeStruct(adjAll,numStatesAll);
nEdges = edgeStruct.nEdges;

% Make Xedge
sharedFeatures = zeros(1,size(Xnode,2));
sharedFeatures(1)=1;
Xedge = UGM_makeEdgeFeatures(Xnode,edgeStruct.edgeEnds,sharedFeatures);
nEdgeFeatures = size(Xedge,2);

% Make edgeMap
f = size(nodeMap,3);
edgeMap = zeros(numStatesAll,numStatesAll,nEdges,nEdgeFeatures,'int32');
%
indicesNonSuperpixelEdges=find(edgeStruct.edgeEnds(:,1)<=nNodes & edgeStruct.edgeEnds(:,2)<=nNodes);
indicesSuperpixelEdgesLayer1=find(edgeStruct.edgeEnds(:,1)<=nNodes & edgeStruct.edgeEnds(:,2)>nNodes & edgeStruct.edgeEnds(:,2)<=endLayer1);
indicesSuperpixelEdgesLayer2=find(edgeStruct.edgeEnds(:,1)<=nNodes & edgeStruct.edgeEnds(:,2)>endLayer1);
%
y(edgeStruct.edgeEnds(indicesSuperpixelEdgesLayer2,2))
%
edgeFeatureWeight=1;
for edgeFeat = 1:nEdgeFeatures
   edgeMap(1,1,indicesNonSuperpixelEdges,edgeFeat) = f+edgeFeatureWeight;
   edgeMap(2,2,indicesNonSuperpixelEdges,edgeFeat) = f+edgeFeatureWeight;
   edgeMap(3,3,indicesNonSuperpixelEdges,edgeFeat) = f+edgeFeatureWeight;
   edgeMap(4,4,indicesNonSuperpixelEdges,edgeFeat) = f+edgeFeatureWeight;
   %
   edgeMap(1,1,indicesSuperpixelEdgesLayer1,edgeFeat)=f+nNodeFeatures+edgeFeatureWeight;
   edgeMap(2,2,indicesSuperpixelEdgesLayer1,edgeFeat)=f+nNodeFeatures+edgeFeatureWeight;
   edgeMap(3,3,indicesSuperpixelEdgesLayer1,edgeFeat)=f+nNodeFeatures+edgeFeatureWeight;
   edgeMap(4,4,indicesSuperpixelEdgesLayer1,edgeFeat)=f+nNodeFeatures+edgeFeatureWeight;
   %
   edgeMap(1,1,indicesSuperpixelEdgesLayer2,edgeFeat)=f+nNodeFeatures+nNodeFeatures+edgeFeatureWeight;
   edgeMap(2,2,indicesSuperpixelEdgesLayer2,edgeFeat)=f+nNodeFeatures+nNodeFeatures+edgeFeatureWeight;
   edgeMap(3,3,indicesSuperpixelEdgesLayer2,edgeFeat)=f+nNodeFeatures+nNodeFeatures+edgeFeatureWeight;
   edgeMap(4,4,indicesSuperpixelEdgesLayer2,edgeFeat)=f+nNodeFeatures+nNodeFeatures+edgeFeatureWeight;
   %
   edgeFeatureWeight=edgeFeatureWeight+1;
   if(edgeFeatureWeight>nNodeFeatures)
       edgeFeatureWeight=1;
   end
   %
end
% 

end

