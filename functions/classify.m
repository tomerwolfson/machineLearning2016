%%%clc; %clear command window

tic; %%% Start measuring program time

positive_reviews = get_files_list('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test2000\neg\');
negative_reviews = get_files_list('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test2000\pos\');

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
min_terms = 0.01*size(review_array, 2); %term thrshold for features, [based on Pang, Lee paper 2001]
[vectors,features] = vectorize_review(review_array, min_terms);
features
% Perform 10 fold cross validation with SVM on the vectors
% use LibSVM implementation
svm_reviews(vectors,labels);
toc %%% Measuring program time in seconds

