function [positive_list, negative_list] = build_words_lists()

negative_list=containers.Map();
data=(readtable('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\words_list\neg_list','ReadVariableNames',false,'ReadRowNames',false));
for i=1:size(data,1)
    word=data{i,1}{1};
    word=porterStemmer(word);
    negative_list(word)=1;
end

positive_list=containers.Map();
data=(readtable('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\words_list\pos_list','ReadVariableNames',false,'ReadRowNames',false));
for i=1:size(data,1)
    word=data{i,1}{1};
    word=porterStemmer(word);
    positive_list(word)=1;
end

end

