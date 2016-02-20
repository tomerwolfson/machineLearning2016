function [ review_array, review_score ] = read_files_contents_and_scores( files_cell_array )
%READ_FILES_CONTENTS_AND_SCORES Summary of this function goes here
%   Detailed explanation goes here
review_array = cell(size(files_cell_array));
review_score = cell(size(files_cell_array));

% go over all the review files,
% extract each review string and store it
fprintf('\nProcessing reviews. Done processing (out of %d):\n', size(files_cell_array,1));
for i = 1:size(files_cell_array,1)
    if (mod(i,25) == 0) || (i == size(files_cell_array,1))
        fprintf('%d ',i);
    end
    if (i == size(files_cell_array,1)) || (mod(i,1000) == 0 && i > 0)
        fprintf('\n');
    end
    myfile = files_cell_array{i};
    
    split_path=strsplit(myfile,'\');%score of review
    filename_cell=split_path(end);
    filename_str=filename_cell{1};
    filename_parts=strsplit(filename_str,{'_','.'});
    score=filename_parts(2);
    score=str2double(score);
    if (score==4 || score==7)
        score=1;
    elseif (score==3 || score==8)
        score=2;
    elseif (score==2 || score==9)
        score=3;
    elseif (score==1 || score==10)
        score=4;
    else
        fprintf(2,'Warning: score %d not in 7-10 or 1-4, for file %s\n', score, myfile);
    end
     
    fid = fopen( myfile);
    s = textscan(fid,'%s','Delimiter','\n');
    mystr = '';
    for mycellindex = 1:size(s{1,1},1)
        mystr = strcat(mystr, s{1,1}{mycellindex});
    end
    fclose(fid);
    review_array{i} = mystr;
    review_score{i}=score;
end

end

