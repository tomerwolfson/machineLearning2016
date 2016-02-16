function [ review_array ] = read_files_contents( files_cell_array )
%READ_FILES_CONTENTS_AND_SCORES Summary of this function goes here
%   Detailed explanation goes here

% Init empty cell array for the results
review_array = cell(length(files_cell_array),1);

% go over all the review files,
% extract each review string and store it
for i = 1:size(files_cell_array,1)
    fprintf(1,'Processing review %d out of %d\n', i, size(files_cell_array,1));
    myfile = files_cell_array{i};
    fid = fopen( myfile );
    s = textscan(fid,'%s','Delimiter','\n');
    mystr = '';
    for mycellindex = 1:size(s{1,1},1)
        mystr = strcat(mystr, s{1,1}{mycellindex});
    end
    fclose(fid);
    review_array{i} = mystr;
end

end

