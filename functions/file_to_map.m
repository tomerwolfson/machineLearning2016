function [words_map] = file_to_map(file_path)

words_map=containers.Map();
fid = fopen(file_path);
data = textscan(fid, '%s');
data = data{1,1};
fclose(fid);

for i=1:size(data,1)
    word=data{i,1};
    word=porterStemmer(word);
    words_map(word)=1;
end

end

