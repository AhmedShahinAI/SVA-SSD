
net = vgg16;

lgraph=layerGraph(net.Layers);


% Remove the last 3 layers. 
layersToRemove = {
    'fc8'
    'prob'
    'output'
    };
lgraph = removeLayers(lgraph, layersToRemove);


% Specify the number of classes the network should classify.
numClasses = 1;
numClassesPlusBackground = numClasses + 1;



%% borrow layers

% Define new classification layers.
newLayers = [
    fullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC')
    softmaxLayer('Name', 'rcnnSoftmax')
    classificationLayer('Name', 'rcnnClassification','Classes',{'houses','background'})
    ];


lgraph=addLayers(lgraph,newLayers);

% Connect the new layers to the network. 
lgraph = connectLayers(lgraph, 'drop7', 'rcnnFC');


analyzeNetwork(lgraph)


fastrcnnlayers=lgraph.Layers;


    [detector, info] = trainFastRCNNObjectDetector(preprocessedTrainingData,fastrcnnlayers,options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);
    
    

 





function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.

scale = targetSize(1:2)./size(data{1},[1 2]);
%size(data{1})
data{1} = imresize((data{1}),targetSize(1:2));

data{2} = bboxresize((data{2}),scale);



%figure;imshow(data{2});
%size(data{2})



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
tform = randomAffine2d('XReflection',true,'YReflection',true,'Scale',[0.5 3]);


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




















