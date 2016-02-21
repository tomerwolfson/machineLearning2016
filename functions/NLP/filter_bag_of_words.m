function [ filtered_bag ] = filter_bag_of_words( ...
    unigram_corpus,unigram_non_corpus,bigram_corpus,bigram_non_corpus,...
    map_sizes, params  )
%FILTER_BAG_OF_WORDS Summary of this function goes here
%   Detailed explanation goes here


selectedheaders=containers.Map();

% words in word bag:
% Filter words (POS or NEG) whose appearance percentage (out of all POS/NEG
% words) is above 0.0001. Keep these words in selectedheaders.
j = 0;
wordkeys = keys(unigram_corpus);
for i=1:size(wordkeys,2)
    app_prec=unigram_corpus(wordkeys{i})/map_sizes(1);
    if (app_prec>=params.word_corpus_thresh)
        j = j + 1;
        selectedheaders(wordkeys{i})=1;
    end
end
fprintf('\nNumber of pos/neg words chosen for the bag of words: %d\n',j);

% words not in word bag:
% Filter words (NOT POS and NOT NEG) whose appearance percentage (out of all NOT POS&NEG
% words) is above 0.05. Keep these words in selectedheaders.
j = 0;
wordkeys = keys(unigram_non_corpus);
for i=1:size(wordkeys,2)
    app_prec=unigram_non_corpus(wordkeys{i})/map_sizes(2);
    if (app_prec>=params.word_not_corpus_thresh)
        j = j + 1;
        selectedheaders(wordkeys{i})=1;
    end
end
fprintf('Number of non-pos/neg words chosen for the bag of words: %d\n',j);

% bigrams in word bag:
% Filter bigrams (2nd POS or NEG) whose appearance percentage (out of all 2nd POS/NEG
% bigrams) is above 0.0001. Keep these words in selectedheaders.
j = 0;
wordkeys = keys(bigram_corpus);
for i=1:size(wordkeys,2)
    app_prec=bigram_corpus(wordkeys{i})/map_sizes(3);
    if (app_prec>=params.bigram_corpus_thresh) % TODO: change to (0.001)^2? 
        j = j + 1;
        selectedheaders(wordkeys{i})=1;
    end
end
fprintf('Number of pos/neg bigrams chosen for the bag of words: %d\n',j);

% bigrams not in word bag:
% Filter bigrams (2nd NOT POS and NOT NEG) whose appearance percentage (out of all 2nd NOT POS&NEG
% bigrams) is above 0.05. Keep these words in selectedheaders.
j = 0;
wordkeys = keys(bigram_non_corpus);
for i=1:size(wordkeys,2)
    app_prec=bigram_non_corpus(wordkeys{i})/map_sizes(4);
    if (app_prec>=params.bigram_not_corpus_thresh)  % TODO: change to (0.05)^2?
        j = j + 1;
        selectedheaders(wordkeys{i})=1;
    end
end
fprintf('Number of non-pos/neg bigrams chosen for the bag of words: %d\n',j);

filtered_bag = keys(selectedheaders);
fprintf('Number of word/bigrams above appearance percentage threshold: %d\n', length(filtered_bag));

end

