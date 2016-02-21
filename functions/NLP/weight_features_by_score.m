function [ featureVector ] = weight_features_by_score( featureVector, review_score )
%WEIGH_FEATURES_BY_SCORE Weights a feature matrix using some review scores.
weights = repmat(cell2mat(review_score),1,size(featureVector,2));
featureVector = featureVector .* weights;
end