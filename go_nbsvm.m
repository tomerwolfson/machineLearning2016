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
load('trained_models\all_filtered_bow_V322908.mat'); % Loads: filtered_bag_of_words
allSNumBi_test = featurize_bigram_nbsvm(filtered_bag_of_words,review_array_test, 1, 1); % featureVector (matrix)

%% Run test function and print results
labels = zeros(length(review_array_test),1); % dummy labels
labels_nbsvm_test = labels2nbsvm_format(labels);

% (2) load trained model
load('trained_models\nbsvm_models\nbsvm_V322908.mat');
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
params.dictsize = length(filtered_bag_of_words);

% (3) classify 
[acc predicted_labels softpred] = testMNBSVM(model, allSNumBi_test, labels_nbsvm_test, params);

% Save the predicted labels as a text file predicted.txt
files_names=cell(length(files),1);
for i = 1:length(files)
    [~,name,ext] = fileparts(files{i});
    strname=strcat(name,ext);
    files_names{i}=strname;
end
len=cellfun(@numel,files_names);
maxLen=max(len);
len=maxLen-len;

fid = fopen('predicted.txt','wt');
for i = 1:length(files)
    fprintf(fid,'%s%s \t%d\n',files_names{i},repmat(' ',1,len(i)),predicted_labels(i));
end
fclose(fid);

end
