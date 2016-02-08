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
review_array ={};

% go over all the review files,
% extract each review string and store it
for i = 1:size(allfiles,1)
    disp(sprintf('Processing review %d out of %d', i, size(allfiles,1)));
    myfile = allfiles{i};
    fid = fopen( myfile);
    s = textscan(fid,'%s','Delimiter','\n');
    mystr = '';
    for mycellindex = 1:size(s{1,1},1)
        mystr = strcat(mystr, s{1,1}{mycellindex});
    end
    fclose(fid);
    review_array{end+1} = mystr;
end
review_array = review_array';


% Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review

%%%features_threshold = 70;
features_threshold = size(allfiles,1)*0.05;
[featureVector, train_features] = featurize_bigram(review_array, features_threshold, 1, 1);%%%%%%
%%%featureVector= featurize_bigram(review_array, features_threshold, 0, 0);
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
negfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test2000\neg\');
posfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test2000\pos\');
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

disp('SVM - Polynomial Kernel');
d = 1;
c = 32;
param = sprintf('-t %d -d %d -c %d -q',1,d,c);
traininglabel = labels;
testlabel = test_labels;
trainingset = featureVector;
testset = testVector;
SVMSModel = svmtrain(traininglabel,trainingset,param);
%classify test:
[Group, accuracy, ~] = svmpredict(testlabel,testset,SVMSModel); %predict without display '-q'
accuracy_SVM = accuracy(1)/100

O1 = NaiveBayes.fit(trainingset,traininglabel,'dist','mn'); % or  'mvmn'
C2 = O1.predict(testset);
error = sum(xor(C2, testlabel));
test_size = size(test_labels, 1);
accuracy_NB = 1 - error/test_size

%################################################################


% % Perform 10 fold cross validation with SVM on the vectors
% % using LibSVM implementation
% disp('SVM - Polynomial Kernel');
% Fresults = [];
% end_pow = 7;
% svm_accuracies = 1:end_pow;
% for d = 1:1
%     for pow = 1:end_pow
%         c = power(2, pow-1);
%         param = sprintf('-t %d -d %d -c %d -q',1,d,c);
%
%         dataset_size = size(review_array, 1);
%         test_size = floor(dataset_size/10);
%         dataset_size = test_size * 10; %round down dataset size
%
%         for i = 1:10
%             randomindices = randperm(dataset_size);
%             randomindices = randomindices(1:(dataset_size-test_size));
%             otherindices = (1:dataset_size)';
%             testsetindex = setdiff(otherindices,randomindices)';
%             trainingsetindex = randomindices ;
%             trainingset = featureVector(trainingsetindex,:);
%             traininglabel = labels(trainingsetindex,:);
%
%             testset = featureVector(testsetindex,:);
%             testlabel = labels(testsetindex,:);
%
%             SVMSModel = svmtrain(traininglabel,trainingset,param);
%             %classify test:
%             [Group, accuracy, ~] = svmpredict(testlabel,testset,SVMSModel); %predict without display '-q'
%             accuracy = accuracy(1)/100;
%             %%%%cMat2 = confusionmat(testlabel,C2);
%             %%%%%%Fresults = [Fresults,F1measureConfusionMatrix(cMat2)];
%             Fresults = [Fresults,accuracy];
%         end
%         svm_accuracies(pow) = mean(Fresults);
%     end
%     %plot graph for all the d's
%     subplot(2,2,d)
%     plot(0:end_pow-1, svm_accuracies);
%     title(['SVM: 25000 reviews polynomial kernel degree: ',num2str(d)]);
%     %%%title(['SVM: 2000 reviews [no stem, no stop-words removal] polynomial kernel degree: ',num2str(d)]);
%     xlabel('log2(C)');
%     ylabel('accuracies');
% end
% %disp(n);
% %fprintf('Accuracy for libSVM classifier = %0.5f\n', mean(Fresults))
% %size(featureVector)%%%%%%%%%%%
% %labels%%%%%%%%
% %featureVector%%%%%%%%%%%%%%
% %size(featureVector)%%%%%%%%%%%%%
%
%
%
% % Perform 10 fold cross validation with Naive Bayes on the vectors
% % using Matlab implementation
% disp('Naive Bayes - Multinomial');
% Fresults = [];
%
% dataset_size = size(review_array, 1);
% test_size = floor(dataset_size/10);
% dataset_size = test_size * 10; %round down dataset size
%
% for i = 1:10
%     randomindices = randperm(dataset_size);
%     randomindices = randomindices(1:(dataset_size-test_size));
%     otherindices = (1:dataset_size)';
%     testsetindex = setdiff(otherindices,randomindices)';
%     trainingsetindex = randomindices ;
%     trainingset = featureVector(trainingsetindex,:);
%     traininglabel = labels(trainingsetindex,:);
%
%     testset = featureVector(testsetindex,:);
%     testlabel = labels(testsetindex,:);
%     O1 = NaiveBayes.fit(trainingset,traininglabel,'dist','mn'); % or  'mvmn'
%     C2 = O1.predict(testset);
%     error = sum(xor(C2, testlabel));
%     accuracy = 1 - error/test_size;
%     %%%%cMat2 = confusionmat(testlabel,C2);
%     %%%%%%Fresults = [Fresults,F1measureConfusionMatrix(cMat2)];
%     Fresults = [Fresults,accuracy];
% end
% %disp(n);
% fprintf('Accuracy for Naive Bayes classifier = %0.5f\n', mean(Fresults))
% %size(featureVector)%%%%%%%%%%%
% %labels%%%%%%%%
% %featureVector%%%%%%%%%%%%%%
% %size(featureVector)%%%%%%%%%%%%%

toc;

