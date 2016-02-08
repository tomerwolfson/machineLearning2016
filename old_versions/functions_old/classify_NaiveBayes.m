%%%clc; %clear command window

tic; %%% Start measuring program time

positive_reviews = get_files_list('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test200\neg\');
negative_reviews = get_files_list('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test200\pos\');

%initialize reviews labels
neg_size = size(negative_reviews,2);
pos_size = size(positive_reviews,2);
labels = [zeros(neg_size,1); ones(pos_size,1)];

review_files = [negative_reviews; positive_reviews];
num_reviews = size(review_files, 1)*size(review_files, 2)

review_array = {};

% go over all the review files,
% extract each review string and store it
for i = 1:num_reviews
    % progress track:
    disp(sprintf('Extracting review %d / %d ', i, num_reviews));
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
min_terms = 7; %%%%0.02*size(review_array, 2); %term threshold for features, [based on Pang, Lee paper 2001]
%%%[vectors,features] = vectorize_review(review_array, min_terms);
review_array = review_array';
vectors = vectorize_review(review_array, n, 0, 0);
%features;%%%%%



% Perform 10 fold cross validation with Naive-Bayes on the vectors
% using Matlab implementation


disp('Naive Bayes - Multinomial');
%10 fold random permutation
Fresults = [];

dataset_size = size(review_array, 2);
test_size = floor(dataset_size/10);
dataset_size = test_size * 10; %round down dataset size

for i = 1:10
    randomindices = randperm(dataset_size);
    randomindices = randomindices(1:(dataset_size-test_size));
    otherindices = (1:dataset_size)';
    testsetindex = setdiff(otherindices,randomindices)';
    trainingsetindex = randomindices ;
    trainingset = vectors(trainingsetindex,:);
    traininglabel = labels(trainingsetindex,:);
    
    testset = vectors(testsetindex,:);
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
%labels%%%%%%%%
%vectors%%%%%%%%%%%%%%
%size(vectors)%%%%%%%%%%%

toc;