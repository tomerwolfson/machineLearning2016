%clc;
clear all;
close all;
addpath(genpath('.'));
%% Load negative and positive examples from train set
tic;
train_set_name = 'svm25k';
posfiles = getAllFiles(['D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset_imdb\test\pos\']);
negfiles = getAllFiles(['D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset_imdb\test\neg\']);
% posfiles = getAllFiles(['test_sets\',train_set_name,'\pos\']);
% negfiles = getAllFiles(['test_sets\',train_set_name,'\neg\']);
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
featureVector = featurize_test_set(filtered_bag_of_words,review_array, 1,1);
featureVector = weight_features_by_score(featureVector , review_score);
filename = sprintf('trained_models\\%s_feature_matrix_V%d.mat',...
    train_set_name,size(featureVector,2));
save_features(filename,featureVector);
filename = sprintf('trained_models\\%s_filtered_bow_V%d.mat',...
    train_set_name,size(featureVector,2));
save(filename,'filtered_bag_of_words');
fprintf('time for featurizing train examples: %f\n', toc);


%% Train SVM model
tic;
%set  train data
traininglabel = labels;
trainingset = featureVector;

% Standardize the train  matrix for the SVM model
%%%%%%%TODO!!!

% model parameters
d = 1;
c = 32;
param = sprintf('-t %d -d %d -c %d -q',1,d,c);
%train svm model
SVMSModel = svmtrain(traininglabel,trainingset,param);
% save model
save(sprintf('trained_models\\svm_models\\svm_V%d.mat',length(filtered_bag_of_words)),'SVMSModel');
fprintf('time for training model: %f\n', toc);


%% Test the SVM model [Optional]
tic;
negfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test200_cornell\neg\');
posfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test200_cornell\pos\');
test_labels = [zeros(size(negfiles,1),1); ones(size(posfiles,1),1)];
testfiles = [negfiles;posfiles];
test_array ={};

% go over all the review files,
% extract each review string and store it
for i = 1:size(testfiles,1)
    if(~mod(i,100))
        disp(sprintf('Processing test review %d out of %d', i, size(testfiles,1)));
    end
    myfile = testfiles{i};
    fid = fopen( myfile);
    s = textscan(fid,'%s','Delimiter','\n');
    mystr = '';
    for mycellindex = 1:size(s{1,1},1)
        mystr = strcat(mystr, s{1,1}{mycellindex});
    end
    fclose(fid);
    test_array{end+1} = mystr;
end
test_array = test_array';

% Perform classification:
% (1) Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review.
train_features=importdata('.\trained_models\chosen_features_svm25k.mat');
testVector = featurize_test_set(train_features, test_array, 1, 1);
% Standardize the test matrix
%%%%%%%TODO!!!

% (2) load trained model
SVMSModel=importdata('.\trained_models\svm_models\SVM_classifier25k.mat');

% (3) classify
testlabel = randi(2,size(testVector,1),1) - 1; % random dummy labels for svmpredict
[Group, accuracy, ~] = svmpredict(testlabel,testset,SVMSModel,'-q');
% (4) compute the accuracy
accuracy_SVM = accuracy(1)/100

fprintf('time for testing model: %f\n', toc);
