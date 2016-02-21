function save_features( filename, featureMatrix )
featureMatrix = sparse(featureMatrix);
save(filename,'featureMatrix');
end

