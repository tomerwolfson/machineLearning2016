function [featureVector,selectedheaderskeys] = featurize_bigram(inputcellarray,review_score,removeStopWords, doStem)
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
stopwords=file_to_map('words_list\english.stop');
neg_words=file_to_map('words_list\neg_list.txt');
pos_words=file_to_map('words_list\pos_list.txt');

% params

bigram_map_bag = containers.Map();
word_map_bag = containers.Map();
bigram_map = containers.Map();
word_map = containers.Map();
total=zeros(1,4);%words_bag, words_no_bag, big_bag, big_no_bag

for i = 1:size(inputcellarray,1)
    fprintf('Parsing %d/%d\n', i, size(inputcellarray,1));
    review = inputcellarray{i};
    review = parse_review_string(review);
    review = lower(review);
    % split the parse review to an array of words
    words=regexp(review,' ','split');
    prev_word = ''; %for bigram insertion
    % go over the words of the review, ordered according to
    % appearance order
    for j = 1:size(words,2)
        in_list=0;
        if (doStem)
            %stem the word - Porter Stemming Algorithm
            word = porterStemmer(cell2mat(words(j)));
        else
            word = (cell2mat(words(j)));
        end
        
        % if the function caller wants to remove stopwords
        if (removeStopWords)
            not_stop_word = ~isKey(stopwords,word);
        else
            not_stop_word = 1;
        end
        
        if (isKey(neg_words, word) || isKey(pos_words, word))
            %wight=floor(0.1*nminFeatures);
            in_list=1;
        end
        % word exists in word-map
        if (not_stop_word & (~strcmp(word,' ')) & (~strcmp(word,'')))
            if (in_list)
                total(1)=total(1)+1;
                if (isKey(word_map_bag, word))
                    word_map_bag(word) = word_map_bag(word)+ 1;
                else
                    word_map_bag(word) = 1;
                end
            else
                total(2)=total(2)+1;
                if (isKey(word_map, word))
                    word_map(word) = word_map(word)+ 1;
                else
                    word_map(word) = 1;
                end
            end
        end
        % add current bigram
        if((~strcmp(prev_word,'')) & (~strcmp(word,' ')) & (~strcmp(prev_word,' ')) & (~strcmp(word,'')) )
            if (isKey(stopwords,word) & isKey(stopwords,prev_word))%the bigram is all stopwords
                prev_word = word;
                continue;
            end
            %check word is not the first in review, and that words are not ' '
            bigram = char(strcat(prev_word,{' '},word));
            % add bigram to map
            if(in_list)
                total(3)=total(3)+1;
                if (isKey(bigram_map_bag, bigram))
                    bigram_map_bag(bigram)=bigram_map_bag(bigram)+1;
                else
                    bigram_map_bag(bigram)=1;
                end
            else
                total(4)=total(4)+1;
                if (isKey(bigram_map, bigram))
                    bigram_map(bigram)=bigram_map(bigram)+1;
                else
                    bigram_map(bigram)=1;
                end
            end 
        end
        prev_word = word;
    end
end

selectedheaders=containers.Map();

%words in word bag
wordkeys = keys(word_map_bag);
for i=1:size(wordkeys,2)
    app_prec=word_map_bag(wordkeys{i})/total(1);
    if (app_prec>=0.001)
        selectedheaders(wordkeys{i})=1;
    end
end

%words not in word bag
wordkeys = keys(word_map);
for i=1:size(wordkeys,2)
    app_prec=word_map(wordkeys{i})/total(2);
    if (app_prec>=0.1)
        selectedheaders(wordkeys{i})=1;
    end
end

%bigrams in word bag
wordkeys = keys(bigram_map_bag);
for i=1:size(wordkeys,2)
    app_prec=bigram_map_bag(wordkeys{i})/total(3);
    if (app_prec>=0.001)
        selectedheaders(wordkeys{i})=1;
    end
end

%bigrams not in word bag
wordkeys = keys(bigram_map);
for i=1:size(wordkeys,2)
    app_prec=bigram_map(wordkeys{i})/total(4);
    if (app_prec>=0.1)
        selectedheaders(wordkeys{i})=1;
    end
end

headers = keys(selectedheaders)

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
    score=review_score(i);
    outputMatrix(i,:) = term_count(review,score{1},headers);
    
    if mod(i,300)==0
        a = sprintf('%d', i);
        disp(a)
    end
    
end

featureVector = outputMatrix;
selectedheaderskeys = keys(selectedheaders);

end

