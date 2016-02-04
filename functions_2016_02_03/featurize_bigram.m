function [featureVector,selectedheaderskeys] = featurize_bigram(inputcellarray, nminFeatures, removeStopWords, doStem)
%
%
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


%
fid = fopen('english.stop');

stopwords = textscan(fid, '%s');
stopwords = stopwords{1,1};
fclose(fid);


% params
% n:minimum appearances of a stem
n=nminFeatures;

word_map = containers.Map();
bigram_map = containers.Map();
prev_word = ''; %for bigram insertion

for i = 1:size(inputcellarray,1)
    fprintf('Parsing %d/%d\n', i, size(inputcellarray,1));
    review = inputcellarray{i};
    review = parse_review_string(review);
    review = lower(review);
    % split the parse review to an array of words
    words=regexp(review,' ','split');
    
    % go over the words of the review, ordered according to
    % appearance order
    for j = 1:size(words,2)
        
        if (doStem)
            %stem the word - Porter Stemming Algorithm
            word = porterStemmer(cell2mat(words(j)));
        else
            word = (cell2mat(words(j)));
        end
        
        % if the function caller wants to remove stopwords
        if (removeStopWords)
            not_stop_word = ~isStopWord(word, stopwords);
        else
            not_stop_word = 1;
        end
        
        % word exists in word-map
        if ( isKey(word_map, word) && not_stop_word )
            word_map(word) = word_map(word)+1;
            % new word encountered
        elseif (not_stop_word & (~strcmp(word,' ')) & (~strcmp(word,'')))
            word_map(word) = 1;
        end
        
        % add current bigram
        if  ( (~strcmp(prev_word,'')) & (~strcmp(word,' ')) & (~strcmp(prev_word,' ')) & (~strcmp(word,'')) )
            %check word is not the first in review, and that words are not ' '
            bigram = char(strcat(prev_word, {' '}, word));
            % add bigram to map
            if (isKey(word_map, bigram))
                word_map(bigram) = word_map(bigram)+1;
            else
                bigram_map(bigram) = 1;
                word_map(bigram) = 1;
            end
        end
        prev_word = word;
        
    end
end

selectedheaders = containers.Map();
wordkeys = keys(word_map);

% extract features
for i=1:size(wordkeys,2)
    if (word_map(wordkeys{i})>=n)
        selectedheaders(wordkeys{i})=1;
    end
end
headers = keys(selectedheaders);

% Iterate over all the reviews and create their vector represenataion,
% the vector coordinates are the features unigrams and bigrams

outputMatrix = zeros(size(inputcellarray,1),length(headers));
for i = 1:size(inputcellarray,1)
    fprintf('Vectorize %d/%d ', i, size(inputcellarray,1));
    review = inputcellarray{i};
    review = parse_review_string(review);
    review = lower(review);
    
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
        review = [review,' ',word];
    end
    outputMatrix(i,:) = term_count(review, headers);
    
    if mod(i,300)==0
        a = sprintf('%d', i);
        disp(a)
    end
      
end

featureVector = outputMatrix;
selectedheaderskeys = keys(selectedheaders);

end

