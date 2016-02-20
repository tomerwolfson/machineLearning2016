function [ featureVector ] = weight_features_by_binarization( featureVector )
%WEIGHT_FEATURES_BY_BINARIZATION Binarize feature vector
featureVector(featureVector > 0) = 1;
end
