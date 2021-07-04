function OutputanchorBoxes=getOptAnchobox(ds,numAnchors)



blds=ds.UnderlyingDatastores{1,2};
inputSize = [300 300 3];
b=ds.UnderlyingDatastores{1, 2}.LabelData(:,1);
GBoxes = vertcat(b{:});
GaspectRatio = GBoxes(:,3) ./ GBoxes(:,4);
Garea = prod(GBoxes(:,3:4),2);
figure
scatter(Garea,GaspectRatio)
xlabel("Box Area")
ylabel("Aspect Ratio (width/height)");
title("Box Area vs. Aspect Ratio For Buildings")


% trainingDataForEstimation = transform(blds ,@(data)preprocessData(data,inputSize));



[OutputanchorBoxes, meanIoU] = estimateAnchorBoxes(blds, numAnchors)

%% Dynamic Caculation

maxNumAnchors =numAnchors;
meanIoU = zeros([maxNumAnchors,1]);
anchorBoxes = cell(maxNumAnchors, 1);
for k = 1:maxNumAnchors
    % Estimate anchors and mean IoU.
    [anchorBoxes{k},meanIoU(k)] = estimateAnchorBoxes(blds,k)  ;  
end


figure
plot(1:maxNumAnchors,meanIoU,'-o')
ylabel("Mean IoU")
xlabel("Number of Anchors")
title("Number of Anchors vs. Mean IoU")




end