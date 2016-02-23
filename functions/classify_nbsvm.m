%clc;
clear all;
close all;
addpath(genpath('.'));
%% Load negative and positive examples from train set
tic;
train_set_name = 'all';
posfiles = getAllFiles(['test_sets\',train_set_name,'\pos\']);
negfiles = getAllFiles(['test_sets\',train_set_name,'\neg\']);
labels = [ones(size(posfiles,1),1); zeros(size(negfiles,1),1)];
allfiles = [posfiles; negfiles];
[review_array, review_score] = read_files_contents_and_scores( allfiles );
clear allfiles negfiles posfiles
fprintf('time for reading train files: %f\n', toc);
%% Convert the textual review into a bag of words
% We select a specific 'bag of words' for the features.
% These features will be the coordinates in the vector representation of
% the review
tic;
[unigram_corpus,unigram_non_corpus,bigram_corpus,bigram_non_corpus,...
    map_sizes] = build_bag_of_words(review_array, 1, 1);
bow_name = ['trained_models\',train_set_name,'_bow_full_stem_stop'];
mkdir(['trained_models\',train_set_name,'_bow_full_stem_stop']);
save_map([bow_name,'\unigram_corpus.mat'],unigram_corpus);
save_map([bow_name,'\unigram_non_corpus.mat'],unigram_non_corpus);
save_map([bow_name,'\bigram_corpus.mat'],bigram_corpus);
save_map([bow_name,'\bigram_non_corpus.mat'],bigram_non_corpus);
fprintf('time for building full bag of words: %f\n', toc);
%% filter bag of words using thresholds for each kind of term
tic;
% % Original thresholds
% params.unigram_corpus_thresh = 0.0001; % 0.0001
% params.unigram_not_corpus_thresh = 0.05; % 0.05
% params.bigram_corpus_thresh = 0.001; % 0.0001
% params.bigram_not_corpus_thresh = 0.05; % 0.05

% % 26293
% params.unigram_corpus_thresh = 0;
% params.unigram_not_corpus_thresh = 0.00001;
% params.bigram_corpus_thresh = 0.00001;
% params.bigram_not_corpus_thresh = 0.00005;

% % about 322908
params.unigram_corpus_thresh = 0;
params.unigram_not_corpus_thresh = 1;
params.bigram_corpus_thresh = 0;
params.bigram_not_corpus_thresh = 1;
filtered_bag_of_words = filter_bag_of_words(unigram_corpus,...
    unigram_non_corpus,bigram_corpus,bigram_non_corpus,map_sizes,params);
fprintf('time for filtering bag of words: %f\n', toc);
% clear bigram_corpus unigram_corpus unigram_non_corpus bigram_non_corpus
%% (old) Convert review to features -  % For old SVM/NB code, and for non binary features
% featureVector = featurize_bigram(filtered_bag_of_words,review_array, 1,1);
% % featureVector = weight_features_by_score(featureVector , review_score);
% filename = sprintf('trained_models\\%s_feature_matrix_V%d.mat',...
%     train_set_name,size(featureVector,2));
% save_features(filename,featureVector);
% filename = sprintf('trained_models\\%s_filtered_bow_V%d.mat',...
%     train_set_name,size(featureVector,2));
% save(filename,'filtered_bag_of_words');
% fprintf('time for featurizing train examples: %f\n', toc);
%% Convert review to features
tic;
allSNumBi = featurize_bigram_nbsvm(filtered_bag_of_words,review_array, 1, 1);
filename = sprintf('trained_models\\%s_feature_matrix_V%d.mat',...
    train_set_name,length(allSNumBi));
save(filename,'allSNumBi');
filename = sprintf('trained_models\\%s_filtered_bow_V%d.mat',...
    train_set_name,length(allSNumBi));
save(filename,'filtered_bag_of_words');
fprintf('time for featurizing train examples: %f\n', toc);
%% Train NBSVM model
tic;
params.C = 1;
params.samplenum = 1;
params.samplerate = 1;
params.Cbisvm = 0.1;
params.a = 1; % params.a is the Laplacian smoothing parameter
params.beta = 0.25; % beta is the interpolation parameter

% this is the exponent used to discount raw counts
% set to 1 to use raw counts f, 
% set to 0 to use indicators \hat{f}
params.testp = 0;
params.trainp = 0;

% cross validation parameters (TODO: NOT NEEDED???????????)
params.CVNUM = 1;
params.doCV = 0;

% % permute data
% p = randperm(size(featureVector,1));
% featureVector = featureVector(p,:);
% labels = labels(p);

% Our code
% allSNumBi = features2nbsvm_format(featureVector); % For old SVM/NB code, and for non binary features
labels_nbsvm = logical(labels'); % labels2nbsvm_format(1-labels);
wordsbi = filtered_bag_of_words;

params.dictsize = length(wordsbi);
params.numcases = length(labels_nbsvm);

cdataset = dataset;
c = 1;
if ~exist('pdataset', 'var') || ~strcmp(cdataset, pdataset)
    pdataset = cdataset;
end
if ~isfield(params, 'doCV')
 params.doCV = 1;
end

trainfuncp = @(allSNumBi, labels, params) trainMNBSVM(allSNumBi, labels, params);
testfuncp = @(model, allSNumBi, labels, params) testMNBSVM(model, allSNumBi, labels, params);

fprintf('Train using dataset l=%d, dictSize=%d, CVNUM=%d\n', ...
    length(allSNumBi), length(filtered_bag_of_words), params.CVNUM)

%randn('state', 0);
%rand('state', 0);

model = trainfuncp(allSNumBi, labels_nbsvm, params);
save(sprintf('trained_models\\nbsvm_models\\nbsvm_V%d.mat',params.dictsize),'model');
fprintf('time for training model: %f\n', toc);
%% Load negative and positive examples from test set
tic;
test_name = 'test_stan';
negfiles_test = getAllFiles(['test_sets\',test_name,'\neg\']);
posfiles_test = getAllFiles(['test_sets\',test_name,'\pos\']);
labels_test = [ones(size(posfiles_test,1),1);zeros(size(negfiles_test,1),1)];
allfiles_test = [posfiles_test;negfiles_test];
[review_array_test, review_score_test] = read_files_contents_and_scores( allfiles_test );
fprintf('time for reading test files: %f\n', toc);
%% Convert the textual review into a feature vector (and locally save the feature vectors) 
tic;
% featureVector = featurize_bigram(filtered_bag_of_words,review_array_test, 1,1); % For old SVM/NB code, and for non binary features
allSNumBi_test = featurize_bigram_nbsvm(filtered_bag_of_words,review_array_test, 1, 1);   
fprintf('time for featurizing test examples: %f\n', toc);
%% Run test function and print results
tic;
labels_nbsvm_test = labels2nbsvm_format(labels_test);
% allSNumBi_test = features2nbsvm_format(featureVector_test); % For old SVM/NB code, and for non binary features

[acc pred softpred] = testfuncp(model, allSNumBi_test, labels_nbsvm_test, params);
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
fprintf('time for calculating test labels: %f\n', toc);

