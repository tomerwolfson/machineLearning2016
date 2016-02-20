function [ allSNumBi ] = features2nbsvm_format( featureVector )
%FEATURS2NBSVM_FORMAT Summary of this function goes here
%   Detailed explanation goes here
allSNumBi = cell(1,size(featureVector,1));
for i = 1:size(featureVector,1)
    allSNumBi{i} = find(featureVector(i,:));
end

end

