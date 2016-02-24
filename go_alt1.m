function go_alt1( test_directory_path )
%GO Main project script for Tomer Wolfson, Sapir Natan and Ofri Galperin
%
%   The script will read the review written in each file in the directory,
%   perform the classification, and save the predicted labels as a text
%   file predicted.txt, in which every row contains a filename and the
%   predicted label (1 or 0, for positive or negative, respectively) for
%   the review in that filename., e.g.:
%	100.txt     1
%	101.txt     0
%	...
%	999.txt     1

% Add current folder and all of its sub-folders (recrursively) to path
addpath(genpath('.'))

% Read the review written in each file
files = getAllFiles(test_directory_path);
test_array = read_files_contents( files );
test_score = cell(length(test_array),1);
for i = 1:length(test_score)
    test_score{i} = 5; % TODO: what to put here?
end

% Perform classification:
% (1) Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review.
train_features=importdata('.\trained_models\chosen_features_svm25k.mat');
testVector = featurize_test_set(train_features, test_array, 1, 1);

%% Normalizing review vectors to range [0,1]
% vec_size=size(testVector);
% vec_count=vec_size(1);
% vec_dim=vec_size(2);
% % get maximal and minimal values for each element in the feature vector
% % (accross all the examples)
% maxVec = max(testVector(:,:));
% minVec = min(testVector(:,:));
% difVec=maxVec-minVec;
% % normalize reviews_vectors
% for vec_num = 1:vec_count;
%     v = testVector(vec_num,:);
%     % normalize to [0,1]
%     for j = 1:vec_dim %check division by 0
%         if (difVec(j) ~= 0)
%             v(j) = ((v(j)-minVec(j))./difVec(j));
%         else
%            v(j)=0; 
%         end
%     end
%     testVector(vec_num,:) = v;
% end
%%
% (2) load trained model
SVMSModel=importdata('.\trained_models\svm_models\SVM_classifier25k.mat');

% (3) classify
testlabel = randi(2,size(testVector,1),1) - 1; % random dummy labels for svmpredict
predicted_labels = svmpredict(testlabel,testVector,SVMSModel,'-q');


% Save the predicted labels as a text file predicted.txt
files_names=cell(length(files),1);
for i = 1:length(files)
    [~,name,ext] = fileparts(files{i});
    strname=strcat(name,ext);
    files_names{i}=strname;
end
len=cellfun(@numel,files_names);
maxLen=max(len);
len=maxLen-len;

fid = fopen('predicted.txt','wt');
for i = 1:length(files)
    fprintf(fid,'%s%s \t%d\n',files_names{i},repmat(' ',1,len(i)),predicted_labels(i));
end
fclose(fid);
end

