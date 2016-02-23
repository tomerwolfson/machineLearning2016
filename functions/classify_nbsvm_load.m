%clc;
clear all;
close all;
addpath(genpath('.'));
%% Load negative and positive examples from train set
train_set_name = 'all';
posfiles = getAllFiles(['test_sets\',train_set_name,'\pos\']);
negfiles = getAllFiles(['test_sets\',train_set_name,'\neg\']);
labels = [ones(size(posfiles,1),1); zeros(size(negfiles,1),1)];
allfiles = [posfiles; negfiles];
[review_array, review_score] = read_files_contents_and_scores( allfiles );
clear allfiles negfiles posfiles
%% Load existing full bad of words
bow_name = ['trained_models\',train_set_name,'_bow_full_stem_stop'];
unigram_corpus = load_map([bow_name,'\unigram_corpus.mat']);
unigram_non_corpus = load_map([bow_name,'\unigram_non_corpus.mat']);
bigram_corpus = load_map([bow_name,'\bigram_corpus.mat']);
bigram_non_corpus = load_map([bow_name,'\bigram_non_corpus.mat']);
map_sizes = [length(keys(unigram_corpus)), ...
             length(keys(unigram_non_corpus)), ...
             length(keys(bigram_corpus)), ...
             length(keys(bigram_non_corpus))];
%% filter bag of words using thresholds for each kind of term
params.unigram_corpus_thresh = 0;     % 3719
params.unigram_not_corpus_thresh = 1; % 0
params.bigram_corpus_thresh = 0; % 16745
params.bigram_not_corpus_thresh = 1;  % 0
filtered_bag_of_words = filter_bag_of_words(unigram_corpus,...
    unigram_non_corpus,bigram_corpus,bigram_non_corpus,map_sizes,params);
%clear bigram_corpus unigram_corpus unigram_non_corpus bigram_non_corpus
%% Convert review to features
featureVector = featurize_bigram(filtered_bag_of_words,review_array, 1, 1);
filename = sprintf('trained_models\\%s_feature_matrix_V%d.mat',...
    train_set_name,size(featureVector,2));
save_features(filename,featureVector);
filename = sprintf('trained_models\\%s_filtered_bow_V%d.mat',...
    train_set_name,size(featureVector,2));
save(filename,'filtered_bag_of_words');
%% Train NBSVM model
% FROM MASTER
params.C = 1;
params.samplenum = 1;
params.samplerate = 1;
params.Cbisvm = 0.1;

% this is the exponent used to discount raw counts
% set to 1 to use raw counts f, 
% set to 0 to use indicators \hat{f}
params.testp = 0;
params.trainp = 0;

% params.a is the Laplacian smoothing parameter
params.a = 1;
% beta is the interpolation parameter
params.beta = 0.25;

% FROM MASTERCV (instead of loading saved data, performs format 
% transformation functions)
% cross validation parameters (TODO: NOT NEEDED???????????)
params.CVNUM = 1;
params.doCV = 0;

% % permute data
% p = randperm(size(featureVector,1));
% featureVector = featureVector(p,:);
% labels = labels(p);

% Our code
labels_nbsvm = logical(labels');%labels2nbsvm_format(1-labels);
allSNumBi = features2nbsvm_format(featureVector);
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
% FROM TRAINTEST
fprintf('Train using dataset l=%d, dictSize=%d, CVNUM=%d\n', ...
    length(allSNumBi), length(filtered_bag_of_words), params.CVNUM)

%randn('state', 0);
%rand('state', 0);

model = trainfuncp(allSNumBi, labels_nbsvm, params);

save(sprintf('trained_models\nbsvm_models\nbsvm_V%d.mat',params.dictsize),'model');
%% Load negative and positive examples from test set
train_set_name = 'test_stan';
negfiles_test = getAllFiles(['test_sets\',train_set_name,'\neg\']);
posfiles_test = getAllFiles(['test_sets\',train_set_name,'\pos\']);
%labels_test = [zeros(size(negfiles_test,1),1); ones(size(posfiles_test,1),1)];
labels_test = [ones(size(posfiles_test,1),1);zeros(size(negfiles_test,1),1)];
%allfiles_test = [negfiles_test;posfiles_test];
allfiles_test = [posfiles_test;negfiles_test];
[review_array_test, review_score_test] = read_files_contents_and_scores( allfiles_test );
%% Convert the textual review into a feature vector (and locally save the feature vectors) 
featureVector_test = featurize_bigram(filtered_bag_of_words,review_array_test, 1, 1);   
%% Run test function and print results
labels_nbsvm_test = labels2nbsvm_format(labels_test);
allSNumBi_test = features2nbsvm_format(featureVector_test);

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
