clc; clear; close all;
addpath MNIST

cur_data = date;  
cur_time = fix(clock);  
date_str = sprintf('%s-%.2d-%.2d-%.2d', cur_data, cur_time(4), cur_time(5), cur_time(6));
filename = ['plain-9-', date_str];
%% activation functions
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

%% initialization
alpha = 0.5;
num_epoch = 400;
batch_size = 1;
L = 10;
mu = 0.9;

m = 5; % input
n = 3; % hidden
k = 2; % output

for i =2:L
    w_T{i} = -1 + 2 * rand(n, n);
    v_w_T{i} = zeros(size(w_T{i}));
end
w_x = -1 + 2 * rand(n, m);
w_k = -1 + 2 * rand(k, n);

v_w_x = zeros(size(w_x));
v_w_k = zeros(size(w_k));

%% train
errEpochArr = [];
for i_epoch = 1:num_epoch
    errArr = [];
    for i = 1:100
        x = rand(5, 1);
        y = zeros(2, 1);
        y(1) = 1;
        
        net_in = w_x * x;
        a{1} = f_in(net_in);
        z_T{1} = a{1};
        for l = 2:L
           z_T{l} = w_T{l} * a{l-1};
           a{l} = f_T(z_T{l});
        end
        net_out = w_k * a{L};
        % Softmax
        pred_exp = exp(net_out);
        predProb = pred_exp ./ repmat(sum(pred_exp, 1), [size(pred_exp, 1), 1]);
        % Cross-entropy
        err = - (1/batch_size) * sum(sum(y .* log(predProb)));
        errArr = [errArr; err];

        delta_out = predProb - y;
        gW_k = delta_out * a{L}';
        % backward
        delta{L} = w_k' * delta_out;
        for l = L:-1:2
           delta{l-1} = (w_T{l}' * delta{l}) .* df_T(z_T{l-1});
        end

        for l = L:-1:2
           g_w_T{l} = delta{l} * a{l-1}';
        end

        g_w_x = (delta{1} .* df_in(net_in)) * x';
        
        %% Gradient check
        epsilon = 0.0001;
    gW_check = zeros(size(w_k));
    w_temp = w_k;
    for i = 1:size(w_temp, 1)
        for j = 1:size(w_temp, 2)
            w_k = w_temp;
            w_k(i, j) = w_k(i, j) + epsilon;

            net_in = w_x * x;
            a{1} = f_in(net_in);
            z_T{1} = a{1};
            for l = 2:L
               z_T{l} = w_T{l} * a{l-1};
               a{l} = f_T(z_T{l});
            end
            net_out = w_k * a{L};
            % Softmax
            pred_exp = exp(net_out);
            predProb = pred_exp ./ repmat(sum(pred_exp, 1), [size(pred_exp, 1), 1]);
            % Cross-entropy
            E1 = - (1/batch_size) * sum(sum(y .* log(predProb)));

            w_k = w_temp;
            w_k(i, j) = w_k(i, j) - epsilon;
            
            net_in = w_x * x;
            a{1} = f_in(net_in);
            z_T{1} = a{1};
            for l = 2:L
               z_T{l} = w_T{l} * a{l-1};
               a{l} = f_T(z_T{l});
            end
            net_out = w_k * a{L};
            % Softmax
            pred_exp = exp(net_out);
            predProb = pred_exp ./ repmat(sum(pred_exp, 1), [size(pred_exp, 1), 1]);
            % Cross-entropy
            E2 = - (1/batch_size) * sum(sum(y .* log(predProb)));

            gW_check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    
    
    
    epsilon = 0.0001;
    gW_check = zeros(size(w_x));
    w_temp = w_x;
    for i = 1:size(w_x, 1)
        for j = 1:size(w_x, 2)
            w_x = w_temp;
            w_x(i, j) = w_x(i, j) + epsilon;

            net_in = w_x * x;
            a{1} = f(net_in);
            z_T{1} = a{1};
            for l = 2:L
               z_T{l} = w_T{l} * a{l-1};
               a{l} = f_T(z_T{l});
            end
            net_out = w_k * a{L};
            % Softmax
            pred_exp = exp(net_out);
            predProb = pred_exp ./ repmat(sum(pred_exp, 1), [size(pred_exp, 1), 1]);
            E1 = - (1/batch_size) * sum(sum(y .* log(predProb)));

            w_x = w_temp;
            w_x(i, j) = w_x(i, j) - epsilon;
            
            net_in = w_x * x;
            a{1} = f(net_in);
            z_T{1} = a{1};
            for l = 2:L
               z_T{l} = w_T{l} * a{l-1};
               a{l} = f_T(z_T{l});
            end
            net_out = w_k * a{L};
            % Softmax
            pred_exp = exp(net_out);
            predProb = pred_exp ./ repmat(sum(pred_exp, 1), [size(pred_exp, 1), 1]);
            E2 = - (1/batch_size) * sum(sum(y .* log(predProb)));

            gW_check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    
    
        % Update
        w_x = w_x - (1/batch_size) * alpha * g_w_x;
        for l = L:-1:2
           w_H{l} = w_H{l} - (1/batch_size) * alpha * g_w_H{l};
           w_T{l} = w_H{l} - (1/batch_size) * alpha * g_w_T{l};
        end
        w_k = w_k - (1/batch_size) * alpha * gW_k;
    end

    %% Test
    pass = 0;
    for i = 1:size(labels_test, 1)/batch_size
        x = images_test(:, (i-1)*batch_size+1:i*batch_size);
        y = labels_test((i-1)*batch_size+1:i*batch_size, :)';

        net_in = w_x * x;
        a{1} = f(net_in);

        for l  = 2:L
           z_H{l} = w_H{l} * a{l-1};
           z_T{l} = w_T{l} * a{l-1};
           aT = f_T(z_T{l});
           a{l} = f_H(z_H{l}) .* aT + a{l-1} .* (1-aT);
        end

        net_out = w_k * a{L};
        pred = f(net_out);
        [~, predCls]=max(pred);
        [~, Cls]=max(y);
        pass = pass + sum(predCls == Cls);
    end
    fprintf('epoch %d, test accuracy:%f, error:%f\n', i_epoch, 100*(pass/size(labels_test, 1)), mean(errArr));
end