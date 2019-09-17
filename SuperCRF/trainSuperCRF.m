clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
pathToTrainingImage1='Output of the single cell classifier'
pathToTrainingImage2='Output of the single cell classifier'
%%%
cellFeaturesAll1 = readtable(pathToTrainingImage1);
cellFeaturesAll1=cellFeaturesAll1(~strcmp(cellFeaturesAll1.classLabeled,'NaN') & ~strcmp(cellFeaturesAll1.classLabeled,'NA'),:);
indNoLym1=~strcmp(cellFeaturesAll1.classPredicted,'l') & ~strcmp(cellFeaturesAll1.classLabeled,'l');
cellFeatures1=cellFeaturesAll1(indNoLym1,:);%No Lyms
%%%
cellFeaturesAll2 = readtable(pathToTrainingImage2);
cellFeaturesAll2=cellFeaturesAll2(~strcmp(cellFeaturesAll2.classLabeled,'NaN') & ~strcmp(cellFeaturesAll2.classLabeled,'NA'),:);
indNoLym2=~strcmp(cellFeaturesAll2.classPredicted,'l') & ~strcmp(cellFeaturesAll2.classLabeled,'l');
cellFeatures2=cellFeaturesAll2(indNoLym2,:);%No Lyms
%%%
pathToSuperpixelImage15x="Output of the superpixel classification (5x)";
pathToSuperpixelImage1SS1="Output of the superpixel classification (1.25x)";
%
pathToSuperpixelImage25x="Output of the superpixel classification (5x)";
pathToSuperpixelImage2SS1="Output of the superpixel classification (1.25x)";
%
%
annotations5x=load("Training annotations for superpixel images");
annotationsSS1=load("Training annotations for superpixel images");
%
annotationMask5X2=annotations5x.s_anno_areas(strcmp({annotations5x.s_anno_areas.name},"Training Image Name1")).mask;
annotationMaskSS12=annotationsSS1.s_anno_areas(strcmp({annotations5x.s_anno_areas.name},"Training Image Name1")).mask;
annotationMask5X1=annotations5x.s_anno_areas(strcmp({annotations5x.s_anno_areas.name},"Training Image Name2")).mask;
annotationMaskSS11=annotationsSS1.s_anno_areas(strcmp({annotations5x.s_anno_areas.name},"Training Image Name2")).mask;
%
superpixelMaskSS11=load(pathToSuperpixelImage1SS1);
superpixelMaskSS11=superpixelMaskSS11.L2;
superpixelMask5x1=load(pathToSuperpixelImage15x);
superpixelMask5x1=superpixelMask5x1.L2;
superpixelMaskSS12=load(pathToSuperpixelImage2SS1);
superpixelMaskSS12=superpixelMaskSS12.L2;
superpixelMask5x2=load(pathToSuperpixelImage25x);
superpixelMask5x2=superpixelMask5x2.L2;
% create the CRFs
[edgeMap1,nodeMap1, Xnode1,Xedge1,edgeStruct1,y1,classP1,classR1,cordsClass1,classR1String,classR1PredictedString,numNodes1 ] = createCRF5x_SS1_sparseDL_annot(cellFeatures1,superpixelMask5x1,superpixelMaskSS11,6,annotationMask5X1,annotationMaskSS11);
[edgeMap2,nodeMap2, Xnode2,Xedge2,edgeStruct2,y2,classP2,classR2,cordsClass2,classR2String,classR2PredictedString,numNodes2 ] = createCRF5x_SS1_sparseDL_annot(cellFeatures2,superpixelMask5x2,superpixelMaskSS12,6,annotationMask5X2,annotationMaskSS12);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%RUN TRAINING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nParams = max([nodeMap1(:);edgeMap1(:)]);
maxIter = 100; % Number of passes through the data set
stepSize = 1e-4;
w = zeros(nParams,1);
nInstances=1;
allLL=[];
for iter = 1:maxIter*nInstances
    i = ceil(rand*nInstances);
    %
    funObj = @(w)UGM_CRF_NLL(w,Xnode1(i,:,:),Xedge1(i,:,:),y1(i,:),nodeMap1,edgeMap1,edgeStruct1,@UGM_Infer_LBP);
    [f1,g1] = funObj(w);
    %
    funObj = @(w)UGM_CRF_NLL(w,Xnode2(i,:,:),Xedge2(i,:,:),y2(i,:),nodeMap2,edgeMap2,edgeStruct2,@UGM_Infer_LBP);
    [f2,g2] = funObj(w);
    %
    g_all=g1+g2;
    f_all=f1+f2;
    %
    allLL=[allLL,f_all];
    fprintf('Iter = %d of %d (fsub = %f)\n',iter,maxIter,f_all);
    
    w = w - stepSize*g_all;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nNodes1=size(Xnode1,3);
[nodePot1,edgePot1] = UGM_CRF_makePotentials(w,Xnode1,Xedge1,nodeMap1,edgeMap1,edgeStruct1,1);
ICMDecoding1 = UGM_Decode_ICM(nodePot1,edgePot1,edgeStruct1);
classCRF1=strings(nNodes1,1);
classCRF1(ICMDecoding1==1)="c";
classCRF1(ICMDecoding1==2)="o";
classCRF1(ICMDecoding1==3)="l";
classCRF1(ICMDecoding1==4)="e";
%
nNodes2=size(Xnode2,3);
[nodePot2,edgePot2] = UGM_CRF_makePotentials(w,Xnode2,Xedge2,nodeMap2,edgeMap2,edgeStruct2,1);
ICMDecoding2 = UGM_Decode_ICM(nodePot2,edgePot2,edgeStruct2);
classCRF2=strings(nNodes2,1);
classCRF2(ICMDecoding2==1)="c";
classCRF2(ICMDecoding2==2)="o";
classCRF2(ICMDecoding2==3)="l";
classCRF2(ICMDecoding2==4)="e";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
accuracySingleCellsCRF1=sum((ICMDecoding1(1:numNodes1)==classR1(1:numNodes1)))/numel(classR1(1:numNodes1))
%%%
accuracySingleCells1=sum((classP1(1:numNodes1)==classR1(1:numNodes1)))/numel(classR1(1:numNodes1))
%%%
accuracySingleCellsCRF2=sum((ICMDecoding2(1:numNodes2)==classR2(1:numNodes2)))/numel(classR2(1:numNodes2))
%%%
accuracySingleCells2=sum((classP2(1:numNodes2)==classR2(1:numNodes2)))/numel(classR2(1:numNodes2))
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawAnnotations( img,strcat('OUTPUT FILE1'),cordsClass(:,1),cordsClass(:,2),classToPredict);
drawAnnotations( img,strcat('OUTPUT FILE2'),cordsClass(:,1),cordsClass(:,2),classCRF);
drawAnnotations( img,strcat('OUTPUT FILE3'),cordsClass(:,1),cordsClass(:,2),class);
