function go( test_directory_path )
%GO Main project script for Tomer Wolfson, Sapir Natan and Ofri Galperin
%
%   The script will read the review written in each file in the directory,
%   perform the classification, and save the predicted labels as a text
%   file predicted.txt, in which every row contains a filename and the
%   predicted label (1 or 0, for positive or negative, respectively) for
%   the review in that filename., e.g.:
%	100.txt     1
%	101.txt     0
%	...
%	999.txt     1

% Add current folder and all of its sub-folfers (recrursively) to path
addpath(genpath('.'))

% Read the review written in each file
files = getAllFiles(test_directory_path);
review_array = read_files_contents( files );
review_score = cell(length(review_array),1); % TODO: How to handle Sapir scores on real test (when no labels, no scores)
for i = 1:length(review_score)
    review_score{i} = 5; % TODO: what to put here?
end

% Perform classification:
% (1) Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review.
[featureVector, train_features] = featurize_bigram(review_array, review_score, 1, 1);%%%%%%

% (2) load trained model
load('trained_models\svm_poly_deg4_c32.mat');

% (3) classify
testlabel = randi(2,size(featureVector,1),1) - 1; % random dummy labels for svmpredict
predicted_labels = svmpredict(testlabel,featureVector,SVMSModel);

% Save the predicted labels as a text file predicted.txt
fid = fopen('predicted.txt','wt');
for i = 1:length(files)
    [~,name,ext] = fileparts(files{i});
    fprintf(fid,'%s\t\t%d\n',[name,ext],predicted_labels(i));
end
fclose(fid);
end

