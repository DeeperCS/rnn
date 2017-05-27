clc; clear; close all;
%% activation functions

f = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df = @(x) f(x) .* (1 - f(x));

f_k = f;  df_k = df;
f_H = f; df_H = df;
f_T = f; df_T = df;
f_out = f;  df_out = df;

delta = @(x,y) x==y;
%% initialization
n = 400; % hidden
m = 2212; % input
k = 2212; % output

depth = 3;

% W_in_T = -1 + 2 * rand(n, m+1);
% W_in_H = -1 + 2 * rand(n, m+1);
W_in_T = normrnd(0, 0.1, [n, m+1]);
W_in_H = normrnd(0, 0.1, [n, m+1]);
for layer =2:depth
    W_R_T{layer} = normrnd(0, 0.1, [n, n+1]);
    W_R_H{layer} = normrnd(0, 0.1, [n, n+1]);
end
W_kc = normrnd(0, 0.1, [k, n+1]);

gW_in_T = zeros(size(W_in_T));
gW_in_H = zeros(size(W_in_H));
gW_R_T = zeros(size(W_R_T));
gW_R_H = zeros(size(W_R_H));
gW_kc = zeros(size(W_kc));

%% train
alpha = 0.01;
epoch_num = 200;

errArr = [];
%Load data
fprintf('Loading data...\n');
itemTotal = 100;
lyrics = {};
for itemNum = 1:itemTotal
    filename = ['mat/lyrics',num2str(itemNum-1),'.mat'];
    load(filename);  % load lyric
    lyrics{itemNum} = lyric;
%     disp(size(lyrics{itemNum}));
end
fprintf('Training...\n');
for i_epoch = 1:epoch_num
    for itemNum = 1:itemTotal    
        x = double(lyrics{itemNum});

        s = {{}};
        s{1}{depth} = zeros(n, 1);
        % Accumulate gradients for every time step    
        gW_in_T = zeros(size(W_in_T));
        gW_in_H = zeros(size(W_in_H));
        gW_R_T = {};
        gW_R_H = {};
        for layer = 2:depth
            gW_R_T{layer} = zeros(size(W_R_T{layer}));
            gW_R_H{layer} = zeros(size(W_R_H{layer}));
        end
        gW_kc = zeros(size(W_kc));

        errMean = [];
        for t = 2:size(x, 1)
            % forward computation
            d = x(t, :)';
            s{t}{1} = s{t-1}{depth};
            for layer = 2:depth
                if layer == 2
                   net_H{t}{layer} = W_in_H * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R_H{layer} * [s{t}{1}; ones([size(s{t}{1}, 2)], 1)];
                   net_T{t}{layer} = W_in_T * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R_T{layer} * [s{t}{1}; ones([size(s{t}{1}, 2)], 1)];
                else
                   net_H{t}{layer} = W_R_H{layer} * [s{t}{layer-1}; ones([size(s{t}{layer-1}, 2)], 1)];
                   net_T{t}{layer} = W_R_T{layer} * [s{t}{layer-1}; ones([size(s{t}{layer-1}, 2)], 1)];
                end
                s{t}{layer} = f_H(net_H{t}{layer}) .* f_T(net_T{t}{layer}) + s{t}{layer-1} .* (1 - f_T(net_T{t}{layer}));
            end
            % Using the last layer in a RHN block as output
            net_k{t} = W_kc * [s{t}{depth}; ones([size(s{t}{depth}, 2), 1])];
            net_out = net_k{t};

            % Softmax and entropy
            net_out = net_out - max(net_out);
            out_exp = exp(net_out);
            y{t} = out_exp ./ sum(out_exp);
    %         err = - d .* log(y{t});
            err = - log( y{t}(find(d==1)) );
%             fprintf('t:%d, y:[%f, %f] net_out:[%f, %f] %f \n', t, min(y{t}), max(y{t}), min(net_out), max(net_out),  err );

            errMean = [errMean; sum(err)];
        end

        delta_next = zeros(n, 1);
        for t = size(x, 1):-1:2
            % compute the gradient of w_kc
            d = x(t, :)';
            delta_K = (y{t} - d);
            gW_kc = gW_kc + delta_K * [s{t}{depth}; ones([size(s{t}{1}, 2), 1])]';

            % backpropagation to s
            delta_s{t}{depth} = (W_kc(:, 1:end-1)' * delta_K) + delta_next;

            for layer = depth:-1:2
                % backpropagation along depth
                delta_s{t}{layer-1} =  ...
                    W_R_H{layer}(:,1:end-1)' * (delta_s{t}{layer} .* df_H(net_H{t}{layer}) .* f_T(net_T{t}{layer}))  +  ...
                    W_R_T{layer}(:,1:end-1)' * (delta_s{t}{layer} .* f_H(net_H{t}{layer}) .* df_T(net_T{t}{layer})) + ...
                    delta_s{t}{layer} .* (1 - f_T(net_T{t}{layer})) - ...
                    W_R_T{layer}(:,1:end-1)' * (delta_s{t}{layer} .* s{t}{layer-1} .* df_T(net_T{t}{layer})); 
                if layer == 2
                    gW_in_T = gW_in_T + ( delta_s{t}{layer} .* ( df_T(net_T{t}{layer}) .* f_H(net_H{t}{layer}) - s{t}{layer-1} .* df_T(net_T{t}{layer}) ) ) * [x(t-1, :), ones([size(x(t-1, :), 1), 1])];
                    gW_in_H = gW_in_H + ( delta_s{t}{layer} .* f_T(net_T{t}{layer}) .* df_H(net_H{t}{layer}) ) * [x(t-1, :), ones([size(x(t-1, :), 1), 1])];     
                    gW_R_T{layer} = gW_R_T{layer} + ( delta_s{t}{layer} .* ( df_T(net_T{t}{layer}) .* f_H(net_H{t}{layer}) - s{t}{layer-1} .* df_T(net_T{t}{layer}) ) ) * [s{t}{1}; ones([size(s{t}{1}, 2)], 1)]';
                    gW_R_H{layer} = gW_R_H{layer} + ( delta_s{t}{layer} .* f_T(net_T{t}{layer}) .* df_H(net_H{t}{layer}) ) * [s{t}{1}; ones([size(s{t}{1}, 2)], 1)]';
                else
                    gW_R_T{layer} = gW_R_T{layer} + ( delta_s{t}{layer} .* ( df_T(net_T{t}{layer}) .* f_H(net_H{t}{layer}) - s{t}{layer-1} .* df_T(net_T{t}{layer}) ) ) * [s{t}{layer-1}; ones([size(s{t}{layer-1}, 2)], 1)]';
                    gW_R_H{layer} = gW_R_H{layer} + ( delta_s{t}{layer} .* f_T(net_T{t}{layer}) .* df_H(net_H{t}{layer}) ) * [s{t}{layer-1}; ones([size(s{t}{layer-1}, 2)], 1)]';
                end 
            end
            delta_next = delta_s{t}{1};
        end

        % Update
        W_in_H = W_in_H - alpha * gW_in_H;
        W_in_T = W_in_T - alpha * gW_in_T;
        for layer = 2:depth
            W_R_T{layer} = W_R_T{layer} - alpha * gW_R_T{layer};
            W_R_H{layer} = W_R_H{layer} - alpha * gW_R_H{layer};
        end
        W_kc = W_kc - alpha * gW_kc;
        

        errArr = [errArr; mean(errMean)];
        fprintf('epoch:%d, item:%d, error:%f \n', i_epoch, itemNum, mean(errMean) );

    end  % Item
    cur_data = date;  
    cur_time = fix(clock);  
    date_str = sprintf('%s-%.2d-%.2d-%.2d', cur_data, cur_time(4), cur_time(5), cur_time(6));
    filename = ['weight-epoch-', num2str(i_epoch), '-time-', date_str];
    save(filename, 'W_in_H', 'W_in_T', 'W_R_T', 'W_R_H', 'W_kc');
end % Epoch
plotHandle = plot(errArr);