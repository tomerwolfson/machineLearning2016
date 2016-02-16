clc;
clear all;
close all;
tic;

%negfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test2000\neg\');
%posfiles = getAllFiles('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\test2000\pos\');
negfiles = getAllFiles('test200\neg\');
posfiles = getAllFiles('test200\pos\');
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


featureVector = featurize_bigram(review_array, review_score, 1, 1);%%%%%%
featureVectorOrig = featureVector;
save('featureVectorn70.dump','featureVector') %save vectors matrix

% Perform 10 fold cross validation with Naive Bayes on the vectors
% using Matlab implementation
disp('Naive Bayes - Multinomial');
Fresults = [];

dataset_size = size(review_array, 1);
test_size = floor(dataset_size/10);
dataset_size = test_size * 10; %round down dataset size

for i = 1:10
    randomindices = randperm(dataset_size);
    randomindices = randomindices(1:(dataset_size-test_size));
    otherindices = (1:dataset_size)';
    testsetindex = setdiff(otherindices,randomindices)';
    trainingsetindex = randomindices ;
    trainingset = featureVector(trainingsetindex,:);
    traininglabel = labels(trainingsetindex,:);
    
    testset = featureVector(testsetindex,:);
    testlabel = labels(testsetindex,:);
    O1 = NaiveBayes.fit(trainingset,traininglabel,'dist','mn'); % or  'mvmn'
    C2 = O1.predict(testset);
    error = sum(xor(C2, testlabel));
    accuracy = 1 - error/test_size;
    Fresults = [Fresults,accuracy];
end
fprintf('Accuracy for Naive Bayes classifier = %0.5f\n', mean(Fresults))

% Normalizing review vectors to range [0,1]
vec_size=size(featureVector);
vec_count=vec_size(1);
vec_dim=vec_size(2);
% get max and min
maxVec = max(featureVector(:,:));
minVec = min(featureVector(:,:));
difVec=maxVec-minVec;
for vec_num = 1:vec_count;%normalize reviews_vectors
    v=featureVector(vec_num,:);
    % normalize to [0,1]
    for j=1:vec_dim %check division by 0
        if (difVec(j)~=0)
            v(j) =((v(j)-minVec(j))./difVec(j));
        end
    end
    featureVector(vec_num,:)=v;
end

% Perform 10 fold cross validation with SVM on the vectors
% using LibSVM implementation
disp('SVM - Polynomial Kernel');
Fresults = [];
d=1;
c = power(2,5);

param = sprintf('-t %d -d %d -c %d -q',1,d,c);

dataset_size = size(review_array, 1);
test_size = floor(dataset_size/10);
dataset_size = test_size * 10; %round down dataset size

for i = 1:10
    randomindices = randperm(dataset_size);
    randomindices = randomindices(1:(dataset_size-test_size));
    otherindices = (1:dataset_size)';
    testsetindex = setdiff(otherindices,randomindices)';
    trainingsetindex = randomindices ;
    trainingset = featureVector(trainingsetindex,:);
    traininglabel = labels(trainingsetindex,:);
    
    testset = featureVector(testsetindex,:);
    testlabel = labels(testsetindex,:);
    
    SVMSModel = svmtrain(traininglabel,trainingset,param);
    %classify test:
    [Group, accuracy, ~] = svmpredict(testlabel,testset,SVMSModel); %predict without display '-q'
    accuracy = accuracy(1)/100;
    Fresults = [Fresults,accuracy];
end

fprintf('Accuracy for SVM classifier = %0.5f\n', mean(Fresults))


toc;

