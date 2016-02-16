clc;
clear all;
close all;
tic;

%############################################%
%### Extract training data:               ###%
%############################################%
negfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\dataset\neg\');
posfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\dataset\pos\');
labels = [zeros(size(negfiles,1),1); ones(size(posfiles,1),1)];

allfiles = [negfiles;posfiles];
[review_array, review_score] = read_files_contents_and_scores( allfiles );

% Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review
[featureVector, train_features] = featurize_bigram(review_array, review_score, 1, 1);%%%%%%
%chosen_features %%%%%%%
featureVectorOrig = featureVector;


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


%############################################%
%### Extract test data:                   ###%
%############################################%
negfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset_cornell\neg\');
posfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset_cornell\pos\');
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
features_threshold = size(allfiles,1)*0.035;
testVector = featurize_test_set(test_array, features_threshold, 1, 1, train_features);
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
%##### Test the classifier on a dataset of 2000 different reviews #########

% Naive Bayes %

%set test and train data
traininglabel = labels;
testlabel = test_labels;
trainingset = featureVector;
testset = testVector;

O1 = NaiveBayes.fit(trainingset,traininglabel,'dist','mn'); % or  'mvmn'
C2 = O1.predict(testset);
error = sum(xor(C2, testlabel));
test_size = size(test_labels, 1);
accuracy_NB = 1 - error/test_size


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
traininglabel = labels;
testlabel = test_labels;
trainingset = featureVector;
testset = testVector;

disp('SVM - Polynomial Kernel');
d = 1;
c = 32;
param = sprintf('-t %d -d %d -c %d -q',1,d,c);
SVMSModel = svmtrain(traininglabel,trainingset,param);
%classify test:
[Group, accuracy, ~] = svmpredict(testlabel,testset,SVMSModel); %predict without display '-q'
accuracy_SVM = accuracy(1)/100

%################################################################

toc;

