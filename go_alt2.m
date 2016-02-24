function go_alt2( test_directory_path )
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

% Add current folder and all of its sub-folders (recrursively) to path
addpath(genpath('.'))

% Read the review written in each file
files = getAllFiles(test_directory_path);
test_array = read_files_contents( files );
test_score = cell(length(test_array),1);
for i = 1:length(test_score)
    test_score{i} = 5; % TODO: what to put here?
end

% Perform classification:
% (1) Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review.
train_features=importdata('.\trained_models\test2000_filtered_bow_V52969.mat');
testVector = featurize_test_set(train_features, test_array, 1, 1);

% (2) load trained model
NBModel=importdata('.\trained_models\nb_models\nb_V52969.mat');

% (3) classify
predicted_labels = NBModel.predict(testVector);

% Save the predicted labels as a text file predicted.txt
fid = fopen('predicted_NB.txt','wt');
for i = 1:length(files)
    [~,name,ext] = fileparts(files{i});
    name=strcat(name,ext);
    maxl=10-length(name);%to check if there is a better way!!!
    fprintf(fid,'%s%s \t%d\n',name,repmat(' ',1,maxl),predicted_labels(i));
end
fclose(fid);

end
