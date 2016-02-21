function [ unigram_corpus,unigram_non_corpus,bigram_corpus,...
    bigram_non_corpus, map_sizes] = build_bag_of_words( ...
        inputcellarray,removeStopWords, doStem)
%BUILD_BAG_OF_WORDS Summary of this function goes here
%   Detailed explanation goes here

stopwords=file_to_map('words_list\english.stop');
neg_words=file_to_map('words_list\neg_list.txt');
pos_words=file_to_map('words_list\pos_list.txt');

% params:
% bigram_corpus -      count the number of shows for each bigram (in all train
%                      samples together), where the bigram 2nd word is in NEG 
%                      or POS corpuses
% unigram_corpus -     count the number of shows for each word (in all train
%                      samples together), where the word is in NEG or POS
%                      corpuses
% bigram_non_corpus -  count the number of shows for each bigram (in all train
%                      samples together), where the bigram 2nd word is NOT in 
%                      NEG or POS corpuses
% unigram_non_corpus - count the number of shows for each word (in all train
%                      samples together), where the word is NOT in NEG or POS
%                      corpuses
unigram_corpus = containers.Map();
unigram_non_corpus = containers.Map();
bigram_corpus = containers.Map();
bigram_non_corpus = containers.Map();
% Size of each map (dictionary)
map_sizes=zeros(1,4); % sizes of each of the maps above by order

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
                map_sizes(1)=map_sizes(1)+1;
                if (isKey(unigram_corpus, word))
                    unigram_corpus(word) = unigram_corpus(word)+ 1;
                else
                    unigram_corpus(word) = 1;
                end
            else
                map_sizes(2)=map_sizes(2)+1;
                if (isKey(unigram_non_corpus, word))
                    unigram_non_corpus(word) = unigram_non_corpus(word)+ 1;
                else
                    unigram_non_corpus(word) = 1;
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
                map_sizes(3)=map_sizes(3)+1;
                if (isKey(bigram_corpus, bigram))
                    bigram_corpus(bigram)=bigram_corpus(bigram)+1;
                else
                    bigram_corpus(bigram)=1;
                end
            else
                map_sizes(4)=map_sizes(4)+1;
                if (isKey(bigram_non_corpus, bigram))
                    bigram_non_corpus(bigram)=bigram_non_corpus(bigram)+1;
                else
                    bigram_non_corpus(bigram)=1;
                end
            end 
        end
        prev_word = word;
    end
end

end

