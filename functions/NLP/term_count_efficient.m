function [ term_count ] = term_count_efficient( review,bow2ind,bow2ind_size )
%TERM_COUNT_EFFICIENT Multinomial Featurizer
%
% takes:
%      inputtext: a long string
%      bow2ind: maps a key (word/bigram) from the bag-of-words to a unique
%               index.
% output:
%      an array of numbers showing how many times each term is repeated in the text
term_count = zeros(1,bow2ind_size);
terms = strsplit(review,' '); % splits the review to a cell array of terms
prev_term = '';
for i = 1:length(terms)
    term = terms{i};
    bigram = [prev_term,' ',term];
    
    if bow2ind.isKey(term)
        term_ind = bow2ind(term);
        term_count(term_ind) = term_count(term_ind) + 1;
    end
    if bow2ind.isKey(bigram) && (i > 1)
        term_ind = bow2ind(bigram);
        term_count(term_ind) = term_count(term_ind) + 1;
    end
    prev_term = term;
end
end

