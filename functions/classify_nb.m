% %clc;
% clear all;
% close all;
% addpath(genpath('.'));
% %% Load negative and positive examples from train set
% tic;
% train_set_name = 'naivebayes25k';
% posfiles = getAllFiles(['D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test2000\pos\']);
% negfiles = getAllFiles(['D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test2000\neg\']);
% % posfiles = getAllFiles(['test_sets\',train_set_name,'\pos\']);
% % negfiles = getAllFiles(['test_sets\',train_set_name,'\neg\']);
% labels = [ones(size(posfiles,1),1); zeros(size(negfiles,1),1)];
% allfiles = [posfiles; negfiles];
% [review_array, review_score] = read_files_contents_and_scores( allfiles );
% clear allfiles negfiles posfiles
% fprintf('time for reading train files: %f\n', toc);
% %% Convert the textual review into a bag of words
% % We select a specific 'bag of words' for the features.
% % These features will be the coordinates in the vector representation of
% % the review
% tic;
% [unigram_corpus,unigram_non_corpus,bigram_corpus,bigram_non_corpus,...
%     map_sizes] = build_bag_of_words(review_array, 1, 1);
% bow_name = ['trained_models\',train_set_name,'_bow_full_stem_stop'];
% mkdir(['trained_models\',train_set_name,'_bow_full_stem_stop']);
% save_map([bow_name,'\unigram_corpus.mat'],unigram_corpus);
% save_map([bow_name,'\unigram_non_corpus.mat'],unigram_non_corpus);
% save_map([bow_name,'\bigram_corpus.mat'],bigram_corpus);
% save_map([bow_name,'\bigram_non_corpus.mat'],bigram_non_corpus);
% fprintf('time for building full bag of words: %f\n', toc);
% %% filter bag of words using thresholds for each kind of term
% tic;
% % Original thresholds
% params.unigram_corpus_thresh = 0.0001; % 0.0001
% params.unigram_not_corpus_thresh = 0.05; % 0.05
% params.bigram_corpus_thresh = 0.001; % 0.0001
% params.bigram_not_corpus_thresh = 0.05; % 0.05
% 
% % % 26293
% % params.unigram_corpus_thresh = 0;
% % params.unigram_not_corpus_thresh = 0.00001;
% % params.bigram_corpus_thresh = 0.00001;
% % params.bigram_not_corpus_thresh = 0.00005;
% 
% % % about 322908 features
% % params.unigram_corpus_thresh = 0;
% % params.unigram_not_corpus_thresh = 1;
% % params.bigram_corpus_thresh = 0;
% % params.bigram_not_corpus_thresh = 1;
% 
% filtered_bag_of_words = filter_bag_of_words(unigram_corpus,...
%     unigram_non_corpus,bigram_corpus,bigram_non_corpus,map_sizes,params);
% fprintf('time for filtering bag of words: %f\n', toc);
% % clear bigram_corpus unigram_corpus unigram_non_corpus bigram_non_corpus
% %% (old) Convert review to features -  % For old SVM/NB code, and for non binary features
% featureVector = featurize_test_set(filtered_bag_of_words,review_array, 1,1);
% featureVector = weight_features_by_score(featureVector , review_score);
% filename = sprintf('trained_models\\%s_feature_matrix_V%d.mat',...
%     train_set_name,size(featureVector,2));
% save_features(filename,featureVector);
% filename = sprintf('trained_models\\%s_filtered_bow_V%d.mat',...
%     train_set_name,size(featureVector,2));
% save(filename,'filtered_bag_of_words');
% fprintf('time for featurizing train examples: %f\n', toc);
% 
% 
% %% Train Naive Bayes model
% tic;
% %set  train data
% traininglabel = labels;
% trainingset = featureVector;
% naiveBayesModel = NaiveBayes.fit(trainingset,traininglabel,'dist','mn'); % or  'mvmn'
% save(sprintf('trained_models\\nb_models\\nb_V%d.mat',length(filtered_bag_of_words)),'naiveBayesModel');
% fprintf('time for training model: %f\n', toc);


%% Test the Naive Bayes model [Optional]
tic;
negfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset_imdb\test\neg\');
posfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset_imdb\test\pos\');
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
train_features=importdata('.\trained_models\naivebayes25k_filtered_bow_V1197.mat');
testVector = featurize_test_set(train_features, test_array, 1, 1);

% (2) load trained model
NBModel=importdata('.\trained_models\nb_models\nb_V1197.mat');

% (3) classify
C2 = NBModel.predict(testVector);

% (4) compute the accuracy
error = sum(xor(C2, test_labels));
test_size = size(test_labels, 1);
accuracy_NB = 1 - error/test_size

fprintf('time for testing model: %f\n', toc);
