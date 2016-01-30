%%%clc; %clear command window

tic; %%% Start measuring program time

positive_reviews = get_files_list('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test200\neg\');
negative_reviews = get_files_list('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test200\pos\');

%initialize reviews labels
neg_size = size(negative_reviews,2);
pos_size = size(positive_reviews,2);
labels = [-ones(neg_size,1); ones(pos_size,1)];

review_files = [negative_reviews; positive_reviews];
num_reviews = size(review_files, 1)*size(review_files, 2)

review_array = {};

% go over all the review files,
% extract each review string and store it
for i = 1:num_reviews
    % progress track:
    %%%disp(sprintf('Extracting review %d / %d ', i, num_reviews));
    %%%disp(review_files(i));
    
    fid = fopen(review_files{i});
    % scan the review, review is in a single line
    review_str = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
       
    review_array{i} = review_str{1};
    %review_array{i}
    %display the review string inisde the cell:
    %%%celldisp(review_array{i})
end

%size(review_array, 1)%%%%%%%%%%%%%%
%size(review_array, 2)%%%%%%%%%%%%%%

% Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review
min_terms = 0.02*size(review_array, 2); %term threshold for features, [based on Pang, Lee paper 2001]
[vectors,features] = vectorize_review(review_array, min_terms);
features
toc;


% Perform 10 fold cross validation with Naive-Bayes on the vectors
% using Matlab implementation

disp('Naive Bayes - Multinomial');
%10 fold random permutation
Fresults = [];

for i = 1:10
    randomindices = randperm(2000);
    randomindices = randomindices(1:1800);
    otherindices = (1:2000)';
    testsetindex = setdiff(otherindices,randomindices)';
    trainingsetindex = randomindices ;
    trainingset = featureVector(trainingsetindex,:);
    traininglabel = labels(trainingsetindex,:);
    
    testset = featureVector(testsetindex,:);
    testlabel = labels(testsetindex,:);
    O1 = NaiveBayes.fit(trainingset,traininglabel,'dist','mn'); % or  'mvmn'
    C2 = O1.predict(testset);
    cMat2 = confusionmat(testlabel,C2);
    Fresults = [Fresults,F1measureConfusionMatrix(cMat2)];
end
%disp(n);
fprintf('F1-measure for Naive Bayes classifier = %0.5f\n', mean(Fresults))

toc;

tic;



disp('Naive Bayes - Bernoulli');
%10 fold validation
Fresults = [];
featureVector = bernoulli(featureVectorOrig);

for i = 1:10
    randomindices = randperm(2000);
    randomindices = randomindices(1:1800);
    otherindices = (1:2000)';
    testsetindex = setdiff(otherindices,randomindices)';
    trainingsetindex = randomindices ;
    trainingset = featureVector(trainingsetindex,:);
    traininglabel = labels(trainingsetindex,:);
    
    testset = featureVector(testsetindex,:);
    testlabel = labels(testsetindex,:);
    O1 = NaiveBayes.fit(trainingset,traininglabel,'dist','mn'); % or  'mvmn'
    C2 = O1.predict(testset);
    cMat2 = confusionmat(testlabel,C2);
    Fresults = [Fresults,F1measureConfusionMatrix(cMat2)];
end
%disp(n);
fprintf('F1-measure for Naive Bayes classifier = %0.5f\n', mean(Fresults))

toc;
