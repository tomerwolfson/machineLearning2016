function [ review_array, review_score ] = read_files_contents_and_scores( files_cell_array )
%READ_FILES_CONTENTS_AND_SCORES Summary of this function goes here
%   Detailed explanation goes here
review_array ={};
review_score ={};

% go over all the review files,
% extract each review string and store it
for i = 1:size(files_cell_array,1)
    disp(sprintf('Processing review %d out of %d', i, size(files_cell_array,1)));
    myfile = files_cell_array{i};
    
    score=strsplit(myfile,'\');%score of review
    score=score(size(score,2));
    score=score{1};
    score=score(size(score,2)-4);
    score=str2num(score);
    if (score==4 || score==7)
        score=1;
    elseif (score==3 || score==8)
        score=2;
    elseif (score==2 || score==9)
        score=3;
    else
        score=4;
    end
     
    fid = fopen( myfile);
    s = textscan(fid,'%s','Delimiter','\n');
    mystr = '';
    for mycellindex = 1:size(s{1,1},1)
        mystr = strcat(mystr, s{1,1}{mycellindex});
    end
    fclose(fid);
    review_array{end+1} = mystr;
    review_score{end+1}=score;
end
review_array = review_array';
review_score = review_score';

end

