testImages = loadMNISTImages('../common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageSize,imageSize,[]);
testLabels = loadMNISTLabels('../common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10
testSize = length(testImages);
preds = zeros(testSize,1);

o1 = zeros(convDim1,convDim1,filterNum1,testSize);
o1Pooled = zeros(outputDim1,outputDim1,filterNum1,testSize);
o2 = zeros(convDim2,convDim2,filterNum2,testSize);
o2Pooled = zeros(outputDim2,outputDim2,filterNum2,testSize);
for i = 1:filterNum1
    o1(:,:,i,:) = convn(testImages,rot90(Wc1(:,:,i),2),'valid') + bc1(i);                                             
end        
o1Pooled = meanPool(o1,poolDim1);        
for i = 1:filterNum2
    for j = 1:filterNum1
        o2(:,:,i,:) = o2(:,:,i,:) + convn(o1Pooled(:,:,j,:),rot90(Wc2(:,:,j,i),2),'valid');
    end
    o2(:,:,i,:) = o2(:,:,i,:) + bc2(i);
end        
o2Pooled = meanPool(o2,poolDim2);        
o2PooledVec = reshape(o2Pooled,[],testSize);
o3 = Wd*o2PooledVec + repmat(bd,[1,testSize]);
    
[~,preds] = max(o3);        

acc = sum(preds'==testLabels)/length(preds);
fprintf('Accuracy is %f\n',acc);
