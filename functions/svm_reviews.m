function svm_reviews(reviews_vectors, labels)
addpath('D:\D\Tomer\Tomer Files\Tel Aviv University\Course_Machine_Learning\project\code\libsvm-3.20\matlab');

vec_size=size(reviews_vectors);
vec_count=vec_size(1);
vec_dim=vec_size(2);

% get max and min
maxVec = max(reviews_vectors(:,:));
minVec = min(reviews_vectors(:,:));
difVec=maxVec-minVec;

for vec_num = 1:vec_count;%normalize reviews_vectors
    v=reviews_vectors(vec_num,:);
    % normalize to [-1,1]
    for j=1:vec_dim
        if difVec(j)~=0
            v(j) =((((v(j)-minVec(j))./(difVec(j))) - 0.5 ) *2);
        end
    end
    reviews_vectors(vec_num,:)=v;
end

%rand matrix
mat_for_rand=[reviews_vectors,labels];
mat_for_rand=mat_for_rand(randperm(size(mat_for_rand,1)),:);
rand_size=size(mat_for_rand);
reviews_vectors=mat_for_rand(:,1:rand_size(2)-1);
labels=(mat_for_rand(:,rand_size(2)));

%values for cross validation:
num_points=floor(vec_count./10) * 10;%25,000
test_size=floor(vec_count/10);%2,500
train_size=num_points-test_size;%25,000-2,500=22,500

end_C = 4;%%%
d = 3;
for count = 1:end_C
    c = power(10,count);% optimal according to tests
    cross_error = 0; %init errors for current C
    set_index = 1; %index of current test set
    for i = 1:10;
        set_index = (test_size*(i-1)) + 1; %move index to beginning of next subset
        %create test matrix:
        test_mat = reviews_vectors(set_index:set_index+test_size-1,:);
        test_labels = labels(set_index:set_index+test_size-1,:);
        %create train matrix:
        train_mat = vertcat(reviews_vectors(1:set_index-1, :), reviews_vectors(set_index+test_size:num_points,:) );
        train_labels = vertcat(labels(1:set_index-1, :), labels(set_index+test_size:num_points,:) );
        %svm error:
        cross_error = cross_error + svmKernels(d,c,train_mat,train_labels,test_mat,test_labels);
    end
    
    final_cross_error = cross_error/10; %set average errors for 10 training sets
    final_accuracy = 1 - final_cross_error;
    final_accu(count) = final_accuracy;
end

%plot graph for all the d's
plot((1:end_C), final_accu(1:end_C));
title(['SVM: 200 reviews polynomial kernel degree: %d',num2str(d)]);
xlabel('log10(C)');
ylabel('accuracies');

    function [MSE] = svmKernels(d,c,train_mat,train_labels,test_mat,test_labels)%create the svm classifier and label the test
        %create classifier:
        param = sprintf('-t %d -d %d -c %d -q',1,d,c);
       %%% param = sprintf('-t %d -c %d -g %d -q',2,c,0.0078125);%%%%%%RBF kernel
        SVMSModel = svmtrain(train_labels,train_mat,param);
        %classify test:
        [Group, accuracy, ~] = svmpredict(test_labels,test_mat,SVMSModel); %predict without display '-q'
        %error clac:
        [test_labels,Group];
        MSE = (100-accuracy(1))/100;
    end
end

