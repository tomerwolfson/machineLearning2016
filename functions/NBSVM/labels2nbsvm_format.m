function [ labels ] = labels2nbsvm_format( labels )
%LABELS2NBSVM_FORMAT Convert labels to NBSVM format
labels = (labels > 0)';
end

