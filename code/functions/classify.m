clc; %clear command window

positive_reviews = get_files_list('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\dataset\pos\');
negative_reviews = get_files_list('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\dataset\neg\');

%initialize reviews labels
neg_size = size(negative_reviews,2);
pos_size = size(positive_reviews,2);
labels = [zeros(neg_size,1); ones(pos_size,1)];

review_files = [negative_reviews; positive_reviews];
num_reviews = size(review_files, 1)*size(review_files, 2);

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
       
    review_array{i} = review_str;
    %display the review string inisde the cell:
    %%%celldisp(review_array{i})
end


% Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review
num_features = 42; %need to decide how many features!!
vectors = vectorize_reviews(review_array, num_features);

% Perform 10 fold cross validation with SVM on the vectors
% use LibSVM implementation