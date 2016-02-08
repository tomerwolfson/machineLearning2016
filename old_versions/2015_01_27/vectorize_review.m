function [review_vectors, chosen_features] = vectorize_review(review_array, num_terms)
%%%
% input:
%   review_array - cell array that contains all the review strings
%   num_terms - the min number of terms of a unigram/bigram in the dataset
%                in order to be selected as a feature
% output:
%   review_vectors - cell array that contains the reviews vectorized form
%   chosen_features - the chosen 'bag of words' used as the features of the reviews
%
% the function receives the strings of all the reviews.
% first, the features of the dataset are extracted. we include the unigrams
% and bigrams that appear more than num_terms times in the review dataset as our features.
% after extracting the features, each review is given its vector
% representaion, according to the selected features.
%%%


% find the features, 'bag of words' of the dataset
% we store for each 'stemmed' unigram and bigram in the data the number of
% occurences. the num_features most popular, will be chosen as the features

unigram_map = containers.Map();
bigram_map = container.Map();

%go over all the review strings
for i = 1:size(review_array, 2)
    review = review_array{i};
    % remove digits, punctuations, and convert to lowercase
    review = parse_review_string(review);
    
    % split the parse review to an array of words
    words = strsplit(review, ' ');
    num_words = size(arr, 2);
    
    % go over the words of the review, ordered according to 
    % appearance order
    prev_word = ''; %for bigram insertion
    for j = 1:num_words
        word = words(j);
        
        %stem the word - Porter Stemming Algorithm
        word = porterStemmer(cell2mat(word));
        
        % check if word is a 'stop word'
        stop_word_flag = is_stop_word(word);
        
        % word exists in unigram map
        if( ~stop_word_flag && isKey(unigram_map, word) )
            unigram_map(word) =  unigram_map(word) + 1;
            % new word encountered
        elseif ( ~stop_word_flag && ~isKey(unigram_map, word) )
            unigram_map(word) = 1;
        end
        
        % add current bigram
        if( ~strcmp(prev_word, ' ') && ~strcmp(prev_word, '') && ~strcmp(word, ' ') && ~strcmp(word, ''))
            %check word is not the first in review, and that words are not ' '
            bigram = strcat(prev_word, {' '}, word); %create the bigram
            % add bigram to map
            if( isKey(bigram_map, bigram) )
                bigram_map(bigram) = bigram_map(bigram) + 1;
            elseif
                bigram_map(bigram) = 1;
            end
        end
        
    end
end

chosen_features = containers.Map();

% extract unigram features
unigram_keyset = keys(unigram_map);
for j = 1:size(unigram_keyset, 2)
    if( unigram_map(unigram_keyset{j}) > num_terms)
        chosen_features(unigram_keyset{j}) = 1;
    end
end

% extract bigram features
bigram_keyset = keys(bigram_map);
for j = 1:size(bigram_keyset, 2)
    if( bigram_map(bigram_keyset{j}) > num_terms)
        chosen_features(bigram_keyset{j}) = 1;
    end
end

% get array of chosen features
feature_grams = keys(chosen_features);


% Iterate over all the reviews and create their vector represenataion,
% the vector coordinates are the features unigrams and bigrams

review_vectors = zeros( size(review_array, 2), length(feature_grams) );
for i = 1:size(review_array, 2)
   review = review_array{i};
   review = parse_review_string(review);
   
   review_words = strsplit(review, ' ');
   stemmed_words = [];
   for j = 1:length(t)
       word = review_words(i);
       word = porterStemmer(cell2mat(word));
       % add stemmed word
       stemmed_words = [stemmed_words, ' ', word];
   end
   
   % return the vector form of the review
   review_vectors(i, :) = feature_stats(stemmed_words, feature_grams);
end

chosen_features = keys(chosen_features);

end




