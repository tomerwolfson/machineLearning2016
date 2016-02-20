clc;
clear all;
close all;
%% Load negative and positive examples from test set
tic;
%negfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test2000\neg\');
%posfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test2000\pos\');
negfiles = getAllFiles('test_sets\test200\neg\');
posfiles = getAllFiles('test_sets\test200\pos\');
labels = [zeros(size(negfiles,1),1); ones(size(posfiles,1),1)];

allfiles = [negfiles;posfiles];
[review_array, review_score] = read_files_contents_and_scores( allfiles );
%% Convert the textual review into a feature vector (and locally save the feature vectors)
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review
[featureVector,selectedheaderskeys] = featurize_bigram(review_array, 1, 1);
featureVector = weight_features_by_score(featureVector, review_score);
% review_score
save('featureVectorn70.mat','featureVector') %save vectors matrix
featureVectorOrig = featureVector;
%% Perform 10 fold cross validation with Naive Bayes on the vectors using Matlab implementation
disp('Naive Bayes - Multinomial');
Fresults = [];

dataset_size = size(review_array, 1);
test_size = floor(dataset_size/10);
dataset_size = test_size * 10; %round down dataset size

% 10-fold cross validation
for i = 1:10
    % choose training and test indices for current fold
    randomindices = randperm(dataset_size);
    randomindices = randomindices(1:(dataset_size-test_size));
    trainingsetindex = randomindices;
    otherindices = (1:dataset_size)';
    testsetindex = setdiff(otherindices,trainingsetindex)';
    % choose training set using the above training indices
    trainingset = featureVector(trainingsetindex,:);
    traininglabel = labels(trainingsetindex,:);
    % choose test set using the above training indices
    testset = featureVector(testsetindex,:);
    testlabel = labels(testsetindex,:);
    % Learn probability distributions from training set using Naive Bayes assumption
    O1 = NaiveBayes.fit(trainingset,traininglabel,'dist','mn'); % or  'mvmn'
    % Predict labels using the learned model
    C2 = O1.predict(testset);
    % calculate performance and store results
    num_errors = sum(xor(C2, testlabel));
    accuracy = 1 - num_errors/test_size;
    fprintf('Accuracy for Naive Bayes classifier (fold %d) = %0.5f\n', i,accuracy)
    Fresults = [Fresults, accuracy];
end
fprintf('Accuracy for Naive Bayes classifier = %0.5f\n', mean(Fresults))
%% Normalizing review vectors to range [0,1]
vec_size=size(featureVector);
vec_count=vec_size(1);
vec_dim=vec_size(2);
% get maximal and minimal values for each element in the feature vector
% (accross all the examples)
maxVec = max(featureVector(:,:));
minVec = min(featureVector(:,:));
difVec=maxVec-minVec;
% normalize reviews_vectors
for vec_num = 1:vec_count;
    v = featureVector(vec_num,:);
    % normalize to [0,1]
    for j = 1:vec_dim %check division by 0
        if (difVec(j) ~= 0)
            v(j) = ((v(j)-minVec(j))./difVec(j));
        end
    end
    featureVector(vec_num,:) = v;
end

%% Perform 10 fold cross validation with SVM on the vectors using LibSVM implementation
disp('SVM - Polynomial Kernel');
Fresults = [];
k = 1; % kernel type (0 -- linear, 1 -- polynomial, 2 -- radial basis function
          %                      3 -- sigmoid, 4 -- precomputed kernel)
d = 1; % polynomial degree
c = power(2,5); % soft svm regularization parameter
n = 1; % Number of cross validation folds

param = sprintf('-t %d -d %d -c %d -q',k,d,c);

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
    Fresults = [Fresults,accuracy];
end
fprintf('Accuracy for SVM classifier = %0.5f\n', mean(Fresults))

toc;

