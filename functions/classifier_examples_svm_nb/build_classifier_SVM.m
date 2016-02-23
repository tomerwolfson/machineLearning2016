clc;
clear all;
close all;


%############################################%
%### Extract training data:               ###%
%############################################%
negfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset\neg\');
posfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\dataset\pos\');
labels = [zeros(size(negfiles,1),1); ones(size(posfiles,1),1)];

allfiles = [negfiles;posfiles];
review_array ={};
review_score ={};

% go over all the review files,
% extract each review string and store it
for i = 1:size(allfiles,1)
    disp(sprintf('Processing review %d out of %d', i, size(allfiles,1)));
    myfile = allfiles{i};
    
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

% Convert the textual review into a feature vector.
% We select a specific 'bag of words' as the features.
% These features will be the coordinates in the vector representation of
% the review


[featureVector, train_features] = featurize_bigram(review_array, review_score, 1, 1);%%%%%%
%chosen_features %%%%%%%
featureVectorOrig = featureVector;


%%%%%%%%%%%%%%%%%%%%delete:
save('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\classifier\train_vectors_svm25k.mat', 'featureVector');
save('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\classifier\chosen_features_svm25k.mat', 'train_features');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% vec_size=size(featureVector);
% vec_count=vec_size(1);
% vec_dim=vec_size(2);
% % get max and min
% maxVec = max(featureVector(:,:));
% minVec = min(featureVector(:,:));
% difVec=maxVec-minVec;
% for vec_num = 1:vec_count;%normalize reviews_vectors
%     v=featureVector(vec_num,:);
%     % normalize to [0,1]
%     v =((v-minVec)./difVec);
%     featureVector(vec_num,:)=v;
% end


% SVM %

% % Normalizing train review vectors to range [0,1]
% vec_size=size(featureVector);
% vec_count=vec_size(1);
% vec_dim=vec_size(2);
% % get max and min
% maxVec = max(featureVector(:,:));
% minVec = min(featureVector(:,:));
% difVec=maxVec-minVec;
% for vec_num = 1:vec_count;%normalize reviews_vectors
%     v=featureVector(vec_num,:);
%     % normalize to [0,1]
%     v =((v-minVec)./difVec);
%     featureVector(vec_num,:)=v;
% end
% 
% % Normalizing test review vectors to range [0,1]
% vec_size=size(testVector);
% vec_count=vec_size(1);
% vec_dim=vec_size(2);
% % get max and min
% maxVec = max(testVector(:,:));
% minVec = min(testVector(:,:));
% difVec=maxVec-minVec;
% 
% for vec_num = 1:vec_count;%normalize reviews_vectors
%     v=testVector(vec_num,:);
%     % normalize to [0,1]
%     v_temp =((v-minVec)./difVec);
%     testVector(vec_num,:)=v_temp;
% end
% v %%%%%%%%%%%%%%%
% minVec %%%%%%%%%%%%
% v-minVec %%%%%%%%%
% difVec %%%%%%%%%%
% v_temp%%%%%%%%%%

%set test and train data
traininglabel = labels;
trainingset = featureVector;

disp('SVM - Polynomial Kernel');
d = 1;
c = 32;
param = sprintf('-t %d -d %d -c %d -q',1,d,c);
%train svm model
SVMSModel = svmtrain(traininglabel,trainingset,param);
%save model
save('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\classifier\SVM_classifier25k', 'SVMSModel');


