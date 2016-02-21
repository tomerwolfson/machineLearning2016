function [ featureMatrix ] = load_features( filename )
load(filename);
featureMatrix = full(featureMatrix);
end

