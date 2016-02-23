clc;
clear all;
close all;
tic;

% Load the Train data:
% 1 - The vectorized train reviews
% 2 - The chosen train features (bigrams & unigrams)
load('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\classifier\train_vectors_nb_25k.mat');
load('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\classifier\chosen_features_nb_25k.mat');



%############################################%
%### Extract test data:                   ###%
%############################################%

%%test_folder_path = 'D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test200';
%%test_folder_path = 'D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset_imdb\test';

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


% Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review
testVector = featurize_test_set(test_array, 1, 1, train_features);

% Naive Bayes %

%set test data
testlabel = test_labels;
testset = testVector;

%load the trained Naive Bayes model
load('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\classifier\NB_classifier_25k');

C2 = O1.predict(testset);
error = sum(xor(C2, testlabel));
test_size = size(test_labels, 1);
accuracy_NB = 1 - error/test_size


%################################################################

toc;

