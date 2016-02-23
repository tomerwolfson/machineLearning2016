clc;
clear all;
close all;
tic;

% Load the Train data:
% 1 - The vectorized train reviews
% 2 - The chosen train features (bigrams & unigrams)
load('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\classifier\train_vectors_svm25k.mat');
load('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\classifier\chosen_features_svm25k.mat');



%############################################%
%### Extract test data:                   ###%
%############################################%
negfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset_imdb\test\neg\');
posfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset_imdb\test\pos\');
test_labels = [zeros(size(negfiles,1),1); ones(size(posfiles,1),1)];

testfiles = [negfiles;posfiles];
test_array ={};

% go over all the review files,
% extract each review string and store it
for i = 1:size(testfiles,1)
    disp(sprintf('Processing test review %d out of %d', i, size(testfiles,1)));
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

%%%features_threshold = 70;
testVector = featurize_test_set(test_array, 1, 1, train_features);
testVectorOrig = testVector;

% vec_size=size(testVector);
% vec_count=vec_size(1);
% vec_dim=vec_size(2);
% % get max and min
% maxVec = max(testVector(:,:));
% minVec = min(testVector(:,:));
% difVec=maxVec-minVec;
% for vec_num = 1:vec_count;%normalize reviews_vectors
%     v=testVector(vec_num,:);
%     % normalize to [0,1]
%     v =((v-minVec)./difVec);
%     testVector(vec_num,:)=v;
% end





%##########################################################################
%##### Test the classifier on a dataset of 200 different reviews #########


% SVM %

% % Normalizing train review vectors to range [0,1]
% vec_size=size(featureVector);
% vec_count=vec_size(1);
% vec_dim=vec_size(2);
% % get max and min
% maxVec = max(featureVector(:,:));
% minVec = min(featureVector(:,:));
% difVec=maxVec-minVec;
% for vec_num = 1:vec_count;%normalize reviews_vectors
%     v=featureVector(vec_num,:);
%     % normalize to [0,1]
%     v =((v-minVec)./difVec);
%     featureVector(vec_num,:)=v;
% end
% 
% % Normalizing test review vectors to range [0,1]
% vec_size=size(testVector);
% vec_count=vec_size(1);
% vec_dim=vec_size(2);
% % get max and min
% maxVec = max(testVector(:,:));
% minVec = min(testVector(:,:));
% difVec=maxVec-minVec;
% 
% for vec_num = 1:vec_count;%normalize reviews_vectors
%     v=testVector(vec_num,:);
%     % normalize to [0,1]
%     v_temp =((v-minVec)./difVec);
%     testVector(vec_num,:)=v_temp;
% end
% v %%%%%%%%%%%%%%%
% minVec %%%%%%%%%%%%
% v-minVec %%%%%%%%%
% difVec %%%%%%%%%%
% v_temp%%%%%%%%%%

%set test and train data
testlabel = test_labels;
testset = testVector;

disp('SVM - Polynomial Kernel');
%load trained SVM model
load('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\classifier\SVM_classifier25k');

%classify test:
[Group, accuracy, ~] = svmpredict(testlabel,testset,SVMSModel); %predict without display '-q'
accuracy_SVM = accuracy(1)/100

%################################################################

toc;

