function [ output_args ] = train_svm_model( train_data_path )
negfiles = getAllFiles('test_sets\test200\neg\');
posfiles = getAllFiles('test_sets\test200\pos\');
labels = [zeros(size(negfiles,1),1); ones(size(posfiles,1),1)];

allfiles = [negfiles;posfiles];
[review_array, review_score] = read_files_contents_and_scores( allfiles );
%% Convert the textual review into a feature vector (and locally save the feature vectors)
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of the review.
featureVector = featurize_bigram(review_array, review_score, 1, 1);
%% Normalizing review vectors to range [0,1]
vec_size=size(featureVector);
vec_count=vec_size(1);
vec_dim=vec_size(2);
% get maximal and minimal values for each element in the feature vector
% (accross all the examples)
maxVec = max(featureVector(:,:));
minVec = min(featureVector(:,:));
difVec=maxVec-minVec;
% normalize reviews_vectors
for vec_num = 1:vec_count;
    v = featureVector(vec_num,:);
    % normalize to [0,1]
    for j = 1:vec_dim %check division by 0
        if (difVec(j) ~= 0)
            v(j) = ((v(j)-minVec(j))./difVec(j));
        end
    end
    featureVector(vec_num,:) = v;
end
%% Train SVM model
disp('SVM - Polynomial Kernel');
Fresults = [];
k = 1; % kernel type (0 -- linear, 1 -- polynomial, 2 -- radial basis function
          %                      3 -- sigmoid, 4 -- precomputed kernel)
d = 4; % polynomial degree
c = power(2,5); % soft svm regularization parameter

param = sprintf('-t %d -d %d -c %d -q',k,d,c);
SVMSModel = svmtrain(labels,featureVector,param);
%% Save model
model_name = sprintf('trained_models\\svm_poly_deg%d_c%d.mat',d,c);
save(model_name,'SVMSModel');


end

