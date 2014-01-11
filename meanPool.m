function pooledFeatures = meanPool(convolvedFeatures,poolDim)

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim/poolDim, ...
        convolvedDim/poolDim,numFilters,numImages);

for i = 1:numImages
    for j = 1:numFilters             
        tmp = conv2(convolvedFeatures(:,:,j,i),ones(poolDim),'valid');
        pooledFeatures(:,:,j,i) = 1./(poolDim^2)*tmp(1:poolDim:end,1:poolDim:end);        
    end
end
