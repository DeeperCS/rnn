clc; clear; close all;
addpath ./cifar10
% epoch 400, test accuracy:96.760000, cross-entropy error:0.000003
cur_data = date;  
cur_time = fix(clock);  
date_str = sprintf('%s-%.2d-%.2d-%.2d', cur_data, cur_time(4), cur_time(5), cur_time(6));
filename = ['highway-9-', date_str];
%% activation functions
% 6.694 s
f = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df = @(x) f(x) .* (1 - f(x));

f_tanh = @(x) (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
df_tanh = @(x) 1 - f_tanh(x) .* f_tanh(x);

relu = @(x) x .* (x>0);
drelu = @(x) (x>0);

h = @(x)2*f(x)-1;
dh = @(x)2*df(x);

f_in = f;  df_in = df;
f_k = f;  df_k = df;
f_T = f; df_T = df;
f_H = h; df_H = dh;

%% initialization
alpha = 0.05;
num_epoch = 400;
batch_size = 1000;
L = 1;
mu = 0.9;

m = 3072; % input
n = 1000; % hidden
k = 10; % output

for i =2:L
%     w_H{i} = -1 + 2 * rand(n, n+1);
%     w_T{i} = -1 + 2 * rand(n, n+1);
    w_H{i} = normrnd(0, 0.1, [n, n+1]);
    w_T{i} = normrnd(0, 0.1, [n, n+1]);
    w_T{i}(:, end) = -2; 
    v_w_H{i} = zeros(size(w_H{i}));
    v_w_T{i} = zeros(size(w_T{i}));
end
% w_x = -1 + 2 * rand(n, m);
% w_k = -1 + 2 * rand(k, n);
w_x = normrnd(0, 0.1, [n, m]);
w_k = normrnd(0, 0.1, [k, n]);

v_w_x = zeros(size(w_x));
v_w_k = zeros(size(w_k));
        
%% train
% Load data
[data_train, label_train, data_test, label_test] = load_cifar10();
images_train_data = double(data_train);
labels_train_data = double(label_train);
label_train_one_hot = zeros(size(labels_train_data, 1), 10);
for i = 1:size(labels_train_data, 1)
    label_train_one_hot(i, labels_train_data(i)+1) = 1;
end

images_test_data = double(data_test);
labels_test_data = double(label_test);
label_test_one_hot = zeros(size(labels_test_data, 1), 10);
for i = 1:size(labels_test_data, 1)
    label_test_one_hot(i, labels_test_data(i)+1) = 1;
end

images_train = images_train_data' ./ 255.0;
labels_train = label_train_one_hot;
images_test = images_test_data' ./ 255.0;
labels_test = label_test_one_hot;

errEpochArr = [];
for i_epoch = 1:num_epoch
    errArr = [];
    for i = 1:size(labels_train, 1)/batch_size
        x = images_train(:, (i-1)*batch_size+1:i*batch_size);
        y = labels_train((i-1)*batch_size+1:i*batch_size, :)';

        net_in = w_x * x;
        a{1} = f_in(net_in);
        for l  = 2:L
           z_H{l} = w_H{l} * [a{l-1}; ones(1, size(a{l-1}, 2))];
           z_T{l} = w_T{l} * [a{l-1}; ones(1, size(a{l-1}, 2))];
           aT = f_T(z_T{l});
           a{l} = f_H(z_H{l}) .* aT + a{l-1} .* (1-aT);
        end
        net_out = w_k * a{L};
        % Softmax
        pred_exp = exp(net_out);
        predProb = pred_exp ./ repmat(sum(pred_exp, 1), [size(pred_exp, 1), 1]);
        % Cross-entropy
        err = - (1/batch_size) * sum(sum(y .* log(predProb)));
%         fprintf('err:%f\n', err);
        errArr = [errArr; err];

        delta_out = predProb - y;
        gW_k = delta_out * a{L}';
        % backward
        delta{L} = w_k' * delta_out;
        for l = L:-1:2
           delta{l-1} = w_H{l}(:,1:end-1)' * (delta{l} .* df_H(z_H{l}) .* f_T(z_T{l}))  +  w_T{l}(:,1:end-1)' * (delta{l} .* f_H(z_H{l}) .* df_T(z_T{l})) + delta{l} .* (1-f_T(z_T{l})) - w_T{l}(:,1:end-1)' * (delta{l} .* a{l-1} .* df_T(z_T{l}));
        end

        for l = L:-1:2
           g_w_H{l} = (delta{l} .* f_T(z_T{l}) .* df_H(z_H{l}) ) * [a{l-1}; ones(1, size(a{l-1}, 2))]';
           g_w_T{l} = (delta{l} .* (df_T(z_T{l}) .* f_H(z_H{l}) - a{l-1} .* df_T(z_T{l}) ) ) * [a{l-1}; ones(1, size(a{l-1}, 2))]';
        end

        g_w_x = (delta{1} .* df_in(net_in)) * x';
        
        % Update (Momentum)
        v_w_x = mu * v_w_x - (1/batch_size) * alpha * g_w_x;
        w_x = w_x + v_w_x;
        for l = L:-1:2
           v_w_H{l} = mu * v_w_H{l} - (1/batch_size) * alpha * g_w_H{l};
           w_H{l} = w_H{l} + v_w_H{l};
           v_w_T{l} = mu * v_w_T{l} - (1/batch_size) * alpha * g_w_T{l};
           w_T{l} = w_T{l} + v_w_T{l};
        end
        v_w_k = mu * v_w_k - (1/batch_size) * alpha * gW_k;
        w_k = w_k + v_w_k;
    end

    %% Test
    pass = 0;
    for i = 1:size(labels_test, 1)/batch_size
        x = images_test(:, (i-1)*batch_size+1:i*batch_size);
        y = labels_test((i-1)*batch_size+1:i*batch_size, :)';

        net_in = w_x * x;
        a{1} = f(net_in);
        for l  = 2:L
           z_H{l} = w_H{l} * [a{l-1}; ones(1, size(a{l-1}, 2))];
           z_T{l} = w_T{l} * [a{l-1}; ones(1, size(a{l-1}, 2))];
           aT = f_T(z_T{l});
           a{l} = f_H(z_H{l}) .* aT + a{l-1} .* (1-aT);
        end
        net_out = w_k * a{L};
        % Softmax
        pred_exp = exp(net_out);
        predProb = pred_exp ./ repmat(sum(pred_exp, 1), [size(pred_exp, 1), 1]);

        [~, predCls]=max(predProb);
        [~, Cls]=max(y);
        pass = pass + sum(predCls == Cls);
    end
    errMean = mean(errArr);
    errEpochArr = [errEpochArr; errMean];
    fprintf('epoch %d, test accuracy:%f, cross-entropy error:%f\n', i_epoch, 100*(pass/size(labels_test, 1)), errMean);
end
save(filename, 'errEpochArr');
disp('finished');