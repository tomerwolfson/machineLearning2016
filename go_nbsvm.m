function go_nbsvm( test_directory_path )
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
review_array_test = read_files_contents( files );

% Perform classification:
% (1) Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review.
load('trained_models\all_filtered_bow_V26293.mat'); % Loads: filtered_bag_of_words
featureVector_test = featurize_bigram(filtered_bag_of_words,review_array_test, 1, 1);   

%% Run test function and print results
% labels = zeros(length(review_array_test),1);
labels = [zeros(length(review_array_test)/2,1); ones(length(review_array_test)/2,1)];
labels_nbsvm_test = labels2nbsvm_format(labels);
allSNumBi_test = features2nbsvm_format(featureVector_test);

% (2) load trained model
load('trained_models\trained_models\nbsvm_models\nbsvm_V26293.mat');
params.C = 1;
params.samplenum = 1;
params.samplerate = 1;
params.Cbisvm = 0.1;
params.testp = 0;
params.trainp = 0;
params.a = 1;
params.beta = 0.25;
params.CVNUM = 1;
params.doCV = 0;

% (3) classify 
[acc predicted_labels softpred] = testfuncp(model, allSNumBi_test, labels_nbsvm_test, params);
% TODO: remove all until acc (including)
nblbltst = labels_nbsvm_test;
fp = sum(nblbltst == 0 & pred == 1);
fn = sum(nblbltst == 1 & pred == 0);
tp = sum(nblbltst == 1 & pred == 1);
tn = sum(nblbltst == 0 & pred == 0);
fprintf('true positives: %d\n',tp);
fprintf('true negatives: %d\n',tn);
fprintf('False positives: %d\n',fp);
fprintf('False negatives: %d\n',fn);
%fprintf('Accuracy: %f\n',acc);
acc

% Save the predicted labels as a text file predicted.txt
fid = fopen('predicted.txt','wt');
for i = 1:length(files)
    [~,name,ext] = fileparts(files{i});
    fprintf(fid,'%s\t\t%d\n',[name,ext],predicted_labels(i));
end
fclose(fid);
end