clc;
clear all;
close all;
addpath(genpath('.'));
%% Load pre-calculated full bag of words
train_set_name = 'train23000';
bow_name = ['trained_models\',train_set_name,'_bow_full_stem_stop'];
unigram_corpus = load_map([bow_name,'\unigram_corpus.mat']);
unigram_non_corpus = load_map([bow_name,'\unigram_non_corpus.mat']);
bigram_corpus = load_map([bow_name,'\bigram_corpus.mat']);
bigram_non_corpus = load_map([bow_name,'\bigram_non_corpus.mat']);
map_sizes = [length(keys(unigram_corpus)), ...
             length(keys(unigram_non_corpus)), ...
             length(keys(bigram_corpus)), ...
             length(keys(bigram_non_corpus))];
%% filter bag of words of size 4732
params.unigram_corpus_thresh = 0;     % 3719
params.unigram_not_corpus_thresh = 1; % 0
params.bigram_corpus_thresh = 0.0001; % 1013
params.bigram_not_corpus_thresh = 1;  % 0
filtered_bag_of_words = filter_bag_of_words(unigram_corpus,...
    unigram_non_corpus,bigram_corpus,bigram_non_corpus,map_sizes,params);
featureVector = featurize_bigram(filtered_bag_of_words,review_array, 1, 1);

filename = sprintf('trained_models\\%s_feature_matrix_V%d.mat',...
    train_set_name,size(featureVector,2));
save_features(filename,featureVector);
%% filter bag of words of size 10187
params.unigram_corpus_thresh = 0;     % 3719
params.unigram_not_corpus_thresh = 1; % 0
params.bigram_corpus_thresh = 0.00002; % 6468
params.bigram_not_corpus_thresh = 1;  % 0
filtered_bag_of_words = filter_bag_of_words(unigram_corpus,...
    unigram_non_corpus,bigram_corpus,bigram_non_corpus,map_sizes,params);
featureVector = featurize_bigram(filtered_bag_of_words,review_array, 1, 1);

filename = sprintf('trained_models\\%s_feature_matrix_V%d.mat',...
    train_set_name,size(featureVector,2));
save_features(filename,featureVector);
%% filter bag of words of size 20464
params.unigram_corpus_thresh = 0;     % 3719
params.unigram_not_corpus_thresh = 1; % 0
params.bigram_corpus_thresh = 0.000008; % 16745
params.bigram_not_corpus_thresh = 1;  % 0
filtered_bag_of_words = filter_bag_of_words(unigram_corpus,...
    unigram_non_corpus,bigram_corpus,bigram_non_corpus,map_sizes,params);
featureVector = featurize_bigram(filtered_bag_of_words,review_array, 1, 1);

filename = sprintf('trained_models\\%s_feature_matrix_V%d.mat',...
    train_set_name,size(featureVector,2));
save_features(filename,featureVector);
%% filter bag of words of size 46163
params.unigram_corpus_thresh = 0;     % 3719
params.unigram_not_corpus_thresh = 1; % 0
params.bigram_corpus_thresh = 0.000003; % 42444
params.bigram_not_corpus_thresh = 1;  % 0
filtered_bag_of_words = filter_bag_of_words(unigram_corpus,...
    unigram_non_corpus,bigram_corpus,bigram_non_corpus,map_sizes,params);
featureVector = featurize_bigram(filtered_bag_of_words,review_array, 1, 1);

filename = sprintf('trained_models\\%s_feature_matrix_V%d.mat',...
    train_set_name,size(featureVector,2));
save_features(filename,featureVector);
