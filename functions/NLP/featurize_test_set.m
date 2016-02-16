function featureVector = featurize_test_set(inputcellarray, nminFeatures, removeStopWords, doStem, classifier_features)

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

% Iterate over all the reviews and create their vector represenataion,
% the vector coordinates are the features unigrams and bigrams

outputMatrix = zeros(size(inputcellarray,1),size(headers, 2));
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
    outputMatrix(i,:) = term_count(review,1, headers);
    
    if mod(i,300)==0
        a = sprintf('%d', i);
        disp(a)
    end
      
end

featureVector = outputMatrix;

end

