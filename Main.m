
% The files are the MATLAB source code for the paper:
% A.I.Shahin, Sultan Almotairi
% SVA-SSD: Saliency Visual Attention Single Shot Detector for Building Detection in Low Contrast High-Resolution Satellite Images
% Submitted to PeerJ Computer Science Journal
% The demo has been organized for peer revieww process. Please contact me if you meet any problems. 
% Email: a.shahin@mu.edu.sa, dr.eng.ahmedshahin@gmail.com


clear
close all
clc

%% Load Dataset

load MRiyadhdataset.mat;
mkdir('buildingimages')
current=cd;
myFolderpath=[current ,'\','buildingimages','\'];
for i=1:height(Riyadhdataset)
Img=Riyadhdataset{i,1};
FILENAME = string(strcat(myFolderpath, num2str(i), '.jpeg'));
imwrite(Img,FILENAME);
end  
S = dir(fullfile(myFolderpath,'*.jpeg'));
N = natsortfiles({S.name});
F = cellfun(@(n)fullfile(myFolderpath,n),N,'uni',0);
imds = imageDatastore(F);
buildingAreasboxs = table(Riyadhdataset(:,2));
buildingAreasboxs.Properties.VariableNames = {'builidng'};
blds=boxLabelDatastore(buildingAreasboxs);
ds_buildings = combine(imds, blds);

%% Split the dataset randomly 80% for training and 20% for testing

varNames = {'imageFilename','builings'};
HDataset=table(ds_buildings.UnderlyingDatastores{1, 1}.Files(:,1),ds_buildings.UnderlyingDatastores{1, 2}.LabelData (:,1),'VariableNames',{'imageFilename','builings'});  

rng(0);
shuffledIndices = randperm(height(HDataset));
idx = floor(0.8 * length(shuffledIndices) );
trainingData = HDataset(shuffledIndices(1:idx),:);
testData = HDataset(shuffledIndices(idx+1:end),:);
imdsTrain = imageDatastore(trainingData{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingData(:,'builings'));
imdsTest = imageDatastore(testData{:,'imageFilename'});
bldsTest = boxLabelDatastore(testData(:,'builings'));
trainingData = combine(imdsTrain,bldsTrain);
testData = combine(imdsTest, bldsTest);


%% Apply Preprocessing Stage and Augmentation

inputSize = [300 300];
augmentedTrainingData = transform(trainingData,@augmentData);
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));


% Visualize N samples of the augmented images.
N=16;
augmentedData = cell(N,1);
for k = 1:N
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',5)

%% Create Our Proposed SVA-SSD Architecture

load preprocesslayer.mat;
% Update Network Input Size
baseNetwork = 'resnet50';
numClasses=1; %% must contains backgrounds
inputSize = [300 300 3];
lgraph = ssdLayers(inputSize,numClasses,baseNetwork);
imageInputSize = [300 300 3];
newinputLayer = [imageInputLayer(imageInputSize,"Name","input")
                 preprocesslayer
                  ]; 
lgraph = replaceLayer(lgraph,"input_1",newinputLayer);

% Saliency Head
net=vgg16;
tranferlayers=net.Layers(2:end-21);
tranferlayers(1, 1).Name='conv1_1_b2';
tranferlayers(2, 1).Name='relu1_1_b2';
tranferlayers(3, 1).Name='conv1_2_b2';
tranferlayers(4, 1).Name='relu1_2_b2';
tranferlayers(5, 1).Name='pool1_b2';
tranferlayers(6, 1).Name='conv2_1_b2';
tranferlayers(7, 1).Name='relu2_1_b2';
tranferlayers(8, 1).Name='conv2_2_b2';
tranferlayers(9, 1).Name='relu2_2_b2';
tranferlayers(10, 1).Name='pool2_b2';
tranferlayers(11, 1).Name='conv3_1_b2';
tranferlayers(12, 1).Name='relu3_1_b2';
tranferlayers(13, 1).Name='conv3_2_b2';
tranferlayers(14, 1).Name='relu3_2_b2';
tranferlayers(15, 1).Name='conv3_3_b2';
tranferlayers(16, 1).Name='relu3_3_b2';
tranferlayers(17, 1).Name='pool3_b2';
tranferlayers(18, 1).Name='conv4_1_b2';
tranferlayers(19, 1).Name='relu4_1_b2';

% add Visual saliency Part

sal=[saliencyLayer2(1,'sal')
     %for mul or add late fusion 
     %convolution2dLayer(2,512,'Name','conv4_3_b2','Padding',[1 1],'Stride',1)%% put it manually 
     %for concatentation
     %convolution2dLayer(2,512,'Name','conv4_3_b2','Padding',1)
     tranferlayers
     convolution2dLayer(2,512,'Name','conv4_3_b2','Padding',[1 1],'Stride',1)%% put it manually 
     %for only Saliency layer with no vgg
     %convolution2dLayer(1,512,'Stride',8,'Name','conv4_3_b2'); 
     softmaxLayer('Name','softmax2')
     reluLayer('Name','rels');    
     ];

lgraph = addLayers(lgraph,sal)
lgraph = connectLayers(lgraph,'preprocessing','sal');%% to replace with sal

%% Fusion 
% multiplicationLayer
% depthConcatenationLayer
fusion =   additionLayer(2,'Name','FUS_1');
lgraph = addLayers(lgraph,fusion);

lgraph = connectLayers(lgraph,'rels','FUS_1/in1');
lgraph = connectLayers(lgraph,'add_7','FUS_1/in2');
lgraph = disconnectLayers(lgraph,'add_7','activation_22_relu');
lgraph = connectLayers(lgraph,"FUS_1","activation_22_relu");


%% Estimate Optimized Anchor BoxSizes
numAnchors=6;
anchorBoxes=getOptAnchobox(trainingData,numAnchors)


anch1 = anchorBoxLayer(anchorBoxes,'Name','relu6_2_anchorbox');
lgraph = replaceLayer(lgraph,'relu6_2_anchorbox',anch1);

anch2 = anchorBoxLayer(anchorBoxes,'Name','relu7_2_anchorbox');
lgraph = replaceLayer(lgraph,'relu7_2_anchorbox',anch2);


%% display Final Network
analyzeNetwork(lgraph);

%% Configure Training Options
%optimizer='sgdm';

optimizer='adam';
      options = trainingOptions(optimizer, ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',0.0001, ...
        'LearnRateDropPeriod', 30, ...
        'LearnRateDropFactor', 0.8, ...
        'MaxEpochs',250, ...
        'VerboseFrequency', 50, ...    
         'ExecutionEnvironment', 'gpu',... 
        'Shuffle','every-epoch');

%% Train the SVA-SSD network.  
[detector,info] = trainSSDObjectDetector(preprocessedTrainingData,lgraph,options);

% display Training Loss
figure
plot(info.TrainingLoss)
grid on
xlabel('Number of Iterations')
ylabel('Training Loss for Each Iteration')


%% Evaluate Performance

preprocessedTestData = transform(testData,@(data)preprocessDatafortest(data,inputSize));
thresh=0.5;
inputSize = [300 300 3];
preprocessedTestData = transform(testData,@(data)preprocessDatafortest(data,inputSize));
numImages = height(preprocessedTestData.UnderlyingDatastores{1, 1}.UnderlyingDatastores{1, 1}.Files);

detectionResults = detect(detector, preprocessedTestData, 'Threshold', 0.5);
[avgap,avgrecall,avgprecision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);


for i = 1 : numImages
   
    I = imread(preprocessedTestData.UnderlyingDatastores{1, 1}.UnderlyingDatastores{1, 1}.Files{i, 1}); 
    groundTruthBboxes=preprocessedTestData.UnderlyingDatastores{1, 1}.UnderlyingDatastores{1, 2}.LabelData{i, 1} ;

    tic; 
    [bboxes, scores] = detect(detector, I,'SelectStrongest',true,'threshold',thresh);
    [bboxes,scores] = selectStrongestBbox(bboxes, scores,'OverlapThreshold',0.01);
    time= toc;
    
    [precision,recall] = bboxPrecisionRecall(bboxes,groundTruthBboxes,thresh);
    overlapRatio = bboxOverlapRatio(bboxes,groundTruthBboxes);
    overlapRatio =overlapRatio(overlapRatio ~=0);
    overlapRatio = overlapRatio(~isnan(overlapRatio));
    AverageOR=mean(mean(overlapRatio));
    AverageOR=AverageOR(~isnan(AverageOR));
    Fscore=(2*precision*recall)/(precision+recall);
    Fscore = Fscore(~isnan(Fscore));
    results.recal{i}=recall;
    results.Fscore{i}=Fscore;
    results.time{i}=time;
    results.Boxes{i} = bboxes;
    results.Scores{i} = scores;
    results.IOU{i} =AverageOR;
    
end 
ap=avgap
recall=mean(cell2mat(results.recal))
Fscore=mean(cell2mat(results.Fscore))
IOU=mean(cell2mat(results.IOU))
time_mean=mean(mean(cell2mat(results.time)))
time_var=var(cell2mat(results.time))









function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
%size(data{1})
data{1} = imresize((data{1}),targetSize(1:2));

data{2} = bboxresize((data{2}),scale);

end



function data = preprocessDatafortest(data,targetSize)
% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
%size(data{1})
data{1} = imresize((data{1}),targetSize(1:2));
data{2}=ceil(uint8(data{2}));
data{2}(data{2}==0)=1;
data{2} = bboxresize((data{2}),scale);
%figure;imshow(data{2});
%size(data{2})
end


function B = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.
B = cell(size(A));

I = A{1};
sz = size(I);
if numel(sz)==3 && sz(3) == 3
    I = jitterColorHSV(I,...
        'Contrast',0.2,...
        'Hue',0,...
        'Saturation',0.1,...
        'Brightness',0.2);
end

% Randomly flip and scale image and rotote.

%tform = randomAffine2d('XReflection',true,'Scale',[0.5 1.5],'Rotation',[-25 25]);
tform = randomAffine2d('XReflection',true,'YReflection',true,'Scale',[0.5 1.5]);


rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');



B{1} = imwarp(I,tform,'OutputView',rout);

%box=ceil(uint16(A{2}));% to assure integer values for bbox

box=ceil((A{2}));% to assure integer values for bbox
box(box==0)=1;
A{2}=box; % replace zero with 1 if found


% Apply same transform to boxes.
[B{2},indices] = bboxwarp(A{2},tform,rout,'OverlapThreshold',0.05);
B{3} = A{3}(indices);

% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end






function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationAccuracy > bestValAccuracy
        valLag = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end


