function file_paths = get_files_list(dir_path)
%%%
% input: directory that contains review files
% ouput: list of the files in the directory
%%%
    directory = dir(dir_path); %matlab directory object
    dirIndexes = [directory.isdir]; % indexes of sub-directories
    file_paths = {directory(~dirIndexes).name}; %get only names of filess
    
    if (~isempty(file_paths))
        % add the full path of the review to each file name
        file_paths = cellfun(@(x) fullfile(dir_path,x), file_paths, 'UniformOutput', false);
    end
end
