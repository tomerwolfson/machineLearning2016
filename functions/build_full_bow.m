clc;
clear all;
close all;
addpath(genpath('.'));
%% Load negative and positive examples from train set
train_set_name = 'train23000';
negfiles = getAllFiles(['test_sets\',train_set_name,'\neg\']);
posfiles = getAllFiles(['test_sets\',train_set_name,'\pos\']);
labels = [zeros(size(negfiles,1),1); ones(size(posfiles,1),1)];

allfiles = [negfiles;posfiles];
[review_array, review_score] = read_files_contents_and_scores( allfiles );
%% Convert the textual review into a bag of words
% We select a specific 'bag of words' for the features.
% These features will be the coordinates in the vector representation of the review
removeStopWords = 1;
doStem = 1;
[unigram_corpus,unigram_non_corpus,bigram_corpus,bigram_non_corpus,...
    map_sizes] = build_bag_of_words(review_array, removeStopWords, doStem);
bow_name = ['trained_models\',train_set_name,'_bow_full_stem_stop'];
mkdir(['trained_models\',train_set_name,'_bow_full_stem_stop']);
save_map([bow_name,'\unigram_corpus.mat'],unigram_corpus);
save_map([bow_name,'\unigram_non_corpus.mat'],unigram_non_corpus);
save_map([bow_name,'\bigram_corpus.mat'],bigram_corpus);
save_map([bow_name,'\bigram_non_corpus.mat'],bigram_non_corpus);