function featureVector = featurize_test_set(inputcellarray, removeStopWords, doStem, classifier_features)

% Receives an array of textual review as the test datase, and an array of features
% chosen by the classifier. The reviews in the araray are converted to vectors according to the classifiers features.
% The ouput vecotrz array will be used as the test set of our trained algorithm.
%
% Inputs:
%       inputcellarray: a cell array with texts as the content of each cell
%       nFeatures: the number of features that we like to see in the vetor
%       removeStopWords: if ==1 it will remove all the stop words
%       doStem: a flag if true porter stemmer will be used
%       classifier_features: chosen features by classifier
% Outputs:
%       featureVector

headers = classifier_features;

% Perform initial parsing of review strings:
% convert all strings to lowercase
fprintf('>> Parsing reviews - converting to lowercase...\n');
inputcellarray = cellfun(@(x) lower(x), inputcellarray, 'UniformOutput', false);
% remove non-character symbols from text
fprintf('>> Parsing reviews - removing non-char symbols...\n');
inputcellarray = cellfun(@(x) parse_review_string(x), inputcellarray, 'UniformOutput', false);
% inputcellarray now contains parsed review strings


% Map of the headers and corresponding indexes
indexed_headers = containers.Map(headers,1:length(headers));% bag of words to index
headers_size = length(headers);

% Create a sparse matrix nXm to hold review vector rep.
% n - the number of reviews
% m - the headers length
% A(n,m) - the number of times header m appears in review m
n = size(inputcellarray,1);
m = headers_size;
mat_row_indexes = [];
mat_col_indexes = [];
mat_vals = [];


% Iterate over reviews and count header occurneces.
% for each review, iterate over every word and check if it coresponds to a
% chosen header. 
% if the word is a header, increment its occurence in the sparse matrix
for(i = 1:n)
    if(~mod(i,10))
        fprintf('Vectorizing review %d/%d \n ', i, n);
    end
    review = inputcellarray{i};
    r=regexp(review,' ','split'); %array containing review words
    review = [];
    %go over words of the review
    for(j = 1:length(r))
        word = char(r(j)); %current word of review
        %stem word
        if (doStem)
            word = porterStemmer(cell2mat(r(j)));
        end
        
        %if word is a header, we increment by 1 its occurence num in
        % the sparse matrix: at row i, and column index_header(word)
        if(isKey(indexed_headers, word))
            col_index = indexed_headers(word);
            % set matrix val - increment occurnence by 1
            mat_vals = [mat_vals 1];
            % update the indexes of the sparse matrix
            mat_row_indexes = [mat_row_indexes i];
            mat_col_indexes = [mat_col_indexes col_index];
        end
    end
end


% Create the full nXm vectorized review matrix
outputMatrix = sparse(mat_row_indexes, mat_col_indexes, mat_vals, n, m);

featureVector = outputMatrix;

end

