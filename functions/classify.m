clc;
clear all;
close all;
tic;


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

features_threshold = 70;
[featureVector headers] = featurize_bigram(review_array, features_threshold, 1, 1);%%%%%%
%%%[featureVector chosen_features]= featurize_bigram(review_array, features_threshold, 0, 0);
%chosen_features %%%%%%%
featureVectorOrig = featureVector;
save('featureVectorn70.dump','featureVector') %save vectors matrix
save('headersn70.dump','headers') %save chosen headers


% Perform 10 fold cross validation with SVM on the vectors
% using LibSVM implementation
disp('SVM Bayes - Polynomial Kernel degree 3');
Fresults = [];
degree = 3;
c = 2^4;
param = sprintf('-t %d -d %d -c %d -q',1,degree,c);

dataset_size = size(review_array, 1);
test_size = floor(dataset_size/10);
dataset_size = test_size * 10; %round down dataset size

for i = 1:10
    randomindices = randperm(dataset_size);
    randomindices = randomindices(1:(dataset_size-test_size));
    otherindices = (1:dataset_size)';
    testsetindex = setdiff(otherindices,randomindices)';
    trainingsetindex = randomindices ;
    trainingset = featureVector(trainingsetindex,:);
    traininglabel = labels(trainingsetindex,:);
    
    testset = featureVector(testsetindex,:);
    testlabel = labels(testsetindex,:);
    
    SVMSModel = svmtrain(traininglabel,trainingset,param);
    %classify test:
    [Group, accuracy, ~] = svmpredict(testlabel,testset,SVMSModel); %predict without display '-q'
    accuracy = accuracy(1)/100;
    %%%%cMat2 = confusionmat(testlabel,C2);
    %%%%%%Fresults = [Fresults,F1measureConfusionMatrix(cMat2)];
    Fresults = [Fresults,accuracy];
end
%disp(n);
fprintf('Accuracy for libSVM classifier = %0.5f\n', mean(Fresults))
%size(featureVector)%%%%%%%%%%%
%labels%%%%%%%%
%featureVector%%%%%%%%%%%%%%
%size(featureVector)%%%%%%%%%%%%%
toc;


% Perform 10 fold cross validation with Naive Bayes on the vectors
% using Matlab implementation
disp('Naive Bayes - Multinomial');
Fresults = [];

dataset_size = size(review_array, 1);
test_size = floor(dataset_size/10);
dataset_size = test_size * 10; %round down dataset size

for i = 1:10
    randomindices = randperm(dataset_size);
    randomindices = randomindices(1:(dataset_size-test_size));
    otherindices = (1:dataset_size)';
    testsetindex = setdiff(otherindices,randomindices)';
    trainingsetindex = randomindices ;
    trainingset = featureVector(trainingsetindex,:);
    traininglabel = labels(trainingsetindex,:);
    
    testset = featureVector(testsetindex,:);
    testlabel = labels(testsetindex,:);
    O1 = NaiveBayes.fit(trainingset,traininglabel,'dist','mn'); % or  'mvmn'
    C2 = O1.predict(testset);
    error = sum(xor(C2, testlabel));
    accuracy = 1 - error/test_size;
    %%%%cMat2 = confusionmat(testlabel,C2);
    %%%%%%Fresults = [Fresults,F1measureConfusionMatrix(cMat2)];
    Fresults = [Fresults,accuracy];
end
%disp(n);
fprintf('Accuracy for Naive Bayes classifier = %0.5f\n', mean(Fresults))
%size(featureVector)%%%%%%%%%%%
%labels%%%%%%%%
%featureVector%%%%%%%%%%%%%%
%size(featureVector)%%%%%%%%%%%%%
toc;


