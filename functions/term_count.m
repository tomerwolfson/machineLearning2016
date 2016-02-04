function output = term_count(inputtext,score,headers)
% Multinomial Featurizer
%
% takes:
%      inputtext: a long string
%      headers: a cell array containing a number of keywords
% output:
%      an array of numbers
%      showing how many times each term is repeated in the text
output = [];
    for i= 1:size(headers,2)
        pattern = headers{i};
        temp = regexp(inputtext, pattern, 'match');   
        output = [output, size(temp,2)*score];
    end

end
