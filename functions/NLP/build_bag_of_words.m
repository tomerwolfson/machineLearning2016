function [bag_of_words] = build_bag_of_words( ...
    inputcellarray,removeStopWords, doStem, params)
%BUILD_BAG_OF_WORDS Summary of this function goes here
%   Detailed explanation goes here

stopwords=file_to_map('words_list\english.stop');
neg_words=file_to_map('words_list\neg_list.txt');
pos_words=file_to_map('words_list\pos_list.txt');

% params:
% bigram_map_bag - count the number of shows for each bigram (in all train
%                  samples together), where the bigram 2nd word is NEG or POS
% word_map_bag   - count the number of shows for each word (in all train
%                  samples together), where the word is NEG or POS
% bigram_map     - count the number of shows for each bigram (in all train
%                  samples together), where the bigram 2nd word is NOT NEG and NOT POS
% word_map       - count the number of shows for each word (in all train
%                  samples together), where the word is NOT NEG and NOT POS
bigram_map_bag = containers.Map();
word_map_bag = containers.Map();
bigram_map = containers.Map();
word_map = containers.Map();
% Size of each map (dictionary)
total=zeros(1,4);% word_map_bag, word_map, bigram_map_bag, bigram_map

fprintf('Calculating bag of words. Total parsed (out of %d):\n', size(inputcellarray,1));
for i = 1:size(inputcellarray,1)
    if (mod(i,25) == 0) || (i == size(inputcellarray,1))
        fprintf('%d ',i);
    end
    if (i == size(inputcellarray,1)) || (mod(i,1000) == 0 && i > 0)
        fprintf('\n');
    end
    
    review = inputcellarray{i};
    review = parse_review_string(review); % Remove html tags, non [A-Za-z] chars; convert to lower case
    % split the parsed review to an array of words
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
        if (not_stop_word && (~strcmp(word,' ')) && (~strcmp(word,'')))
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
        % add current bigram:
        % check word is not the first in review, and that words are not ' '
        if((~strcmp(prev_word,'')) && (~strcmp(word,' ')) && (~strcmp(prev_word,' ')) && (~strcmp(word,'')) ),
            if (isKey(stopwords,word) && isKey(stopwords,prev_word))%the bigram is all stopwords
                prev_word = word;
                continue;
            end
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

% words in word bag:
% Filter words (POS or NEG) whose appearance percentage (out of all POS/NEG
% words) is above 0.0001. Keep these words in selectedheaders.
j = 0;
wordkeys = keys(word_map_bag);
for i=1:size(wordkeys,2)
    app_prec=word_map_bag(wordkeys{i})/total(1);
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
wordkeys = keys(word_map);
for i=1:size(wordkeys,2)
    app_prec=word_map(wordkeys{i})/total(2);
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
wordkeys = keys(bigram_map_bag);
for i=1:size(wordkeys,2)
    app_prec=bigram_map_bag(wordkeys{i})/total(3);
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
wordkeys = keys(bigram_map);
for i=1:size(wordkeys,2)
    app_prec=bigram_map(wordkeys{i})/total(4);
    if (app_prec>=params.bigram_not_corpus_thresh)  % TODO: change to (0.05)^2?
        j = j + 1;
        selectedheaders(wordkeys{i})=1;
    end
end
fprintf('Number of non-pos/neg bigrams chosen for the bag of words: %d\n',j);

bag_of_words = keys(selectedheaders);
fprintf('Number of word/bigrams above appearance percentage threshold: %d\n', length(bag_of_words));

end

