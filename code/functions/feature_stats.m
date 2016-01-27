function review_vec = feature_stats(words, features)
%%%
% input: the word of the review, the feature bigrams and unigrams
% ouput: a vector rep. of the review. each coordinate contains the numbers
% of terms the specified feature appears in the review.
%%%

%column vector
review_vec = zeros(length(features), 1);

for i=1:length(features)
    % get all the appearances of the current feature in the review
    appearances = regexp(words, features(i), 'match');
    review_vec(i) = length(appearances);
end

%turn into row vector
review_vec = review_vec';

end