function [featureVector] = featurize_bigram( ...
    bag_of_words, inputcellarray,removeStopWords, doStem)
% featureVector = featurize_bigram(inputcellarray, nminFeatures, removeStopWords, doStem)
%
% Takes an input cell array in which each cell is a review or text and
% outputs a feature vector with a number of features (nminFeatures is the
% number of times that a feature should be presented in the corpus to be
% included in the feature vector.)
%
% The approach is basically a bag-of-word but we also add bigrams to the
% feature vector too
%
% removeStopWords is a flag, if it is true it will remove the stop words.
%
% Inputs:
%       inputcellarray: a cell array with texts as the content of each cell
%       nFeatures: the number of features that we like to see in the vetor
%       removeStopWords: if ==1 it will remove all the stop words
%       doStem: a flag if true porter stemmer will be used
% Outputs:
%       featureVector

% Test case:
% inputcellarray = {' MATLAB desktop keyboard shortcuts, such as Ctrl+S,  are now customizable.';' To customize keyboard shortcuts, use Preferences. From there, you can also  restore previous default settings by following the steps outlined in Help.'}
% nminFeatures = 1;
% removeStopWords = 0;
% doStem =1;
% featureVector = featurize_bigram(inputcellarray, nminFeatures, removeStopWords, doStem)


% Iterate over all the reviews and create their vector represenataion,
% the vector coordinates are the features unigrams and bigrams
outputMatrix = zeros(size(inputcellarray,1),length(bag_of_words));

% Create bag-of-words order (for the feature vector)
bow2ind = containers.Map(bag_of_words,1:length(bag_of_words));% bag of words to index
bow2ind_size = length(bag_of_words);

fprintf('\nVectorize reviews. Done vectorizing (out of %d):\n', size(inputcellarray,1));
for i = 1:size(inputcellarray,1)
    if (mod(i,25) == 0) || (i == size(inputcellarray,1))
        fprintf('%d ',i);
    end
    if (i == size(inputcellarray,1)) || (mod(i,1000) == 0 && i > 0)
        fprintf('\n');
    end
    
    review = inputcellarray{i};
    review = parse_review_string(review);
    
    r=regexp(review,' ','split');
    review = [];
    for j =1:size(r,2)
        if (doStem)
            % add stemmed word
            word = porterStemmer(cell2mat(r(j)));
        else
            word = (cell2mat(r(j)));
        end
        %stemmed_review
        review = [review,' ',word]; %#ok<AGROW>
    end
    % outputMatrix(i,:) = term_count(review,bag_of_words);
    outputMatrix(i,:) = term_count_efficient(review,bow2ind,bow2ind_size);
end

featureVector = outputMatrix;
end

