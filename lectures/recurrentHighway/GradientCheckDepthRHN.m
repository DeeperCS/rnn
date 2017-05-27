clc; clear; close all;
%% activation functions

f = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df = @(x) f(x) .* (1 - f(x));

h = @(x)2*f(x)-1;
dh = @(x)2*df(x);

h = @(x)x;
dh = @(x)1;

f_k = f;  df_k = df;
f_H = f; df_H = df;
f_T = f; df_T = df;
f_out = f;  df_out = df;

delta = @(x,y) x==y;
%% initialization
n = 8; % hidden
m = 7; % input
k = 7; % output

depth = 3;

W_in_T = -1 + 2 * rand(n, m+1);
W_in_H = -1 + 2 * rand(n, m+1);
for layer =2:depth
    W_R_T{layer} = -1 + 2 * rand(n, n+1);
    W_R_H{layer} = -1 + 2 * rand(n, n+1);
end
W_kc = -1 + 2 * rand(k, n+1);

gW_in_T = zeros(size(W_in_T));
gW_in_H = zeros(size(W_in_H));
gW_R_T = zeros(size(W_R_T));
gW_R_H = zeros(size(W_R_H));
gW_kc = zeros(size(W_kc));

%% train
alpha = 0.5;

trainLength = 10000;
errArr = [];
% plotHandle = plot(errArr);
for tt = 1:trainLength
    % Generate data
    [input, target, str] = embeded_reber_gen();
%     input = input(1:3, :);
    x = input;
    
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
        y{t} = f_k(net_k{t});
        y_k = y{t};
        err = 0.5 * (y_k - d)' * (y_k - d);
        errMean = [errMean; err];
    end
    
    delta_next = zeros(n, 1);
    for t = size(x, 1):-1:2
        % compute the gradient of w_kc
        d = x(t, :)';
        delta_K = (y{t} - d) .* df_k(net_k{t});
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
            
%             gW_R_T{layer} = gW_R_T{layer} + ( delta_s{t}{layer} .* ( df_T(net_T{t}{layer}) .* f_H(net_H{t}{layer}) - s{t}{layer-1} .* df_T(net_T{t}{layer}) ) ) * [s{t}{layer-1}; ones([size(s{t}{layer-1}, 2)], 1)]';
%             gW_R_H{layer} = gW_R_H{layer} + ( delta_s{t}{layer} .* f_T(net_T{t}{layer}) .* df_H(net_H{t}{layer}) ) * [s{t}{layer-1}; ones([size(s{t}{layer-1}, 2)], 1)]';
%             
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
    
    %% Gradient Check
    s = {{}};
    s{1}{depth} = zeros(n, 1);
    % Accumulate gradients for every time step
    temp = W_kc;
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            errMean = [];
            W_kc = temp;
            W_kc(i, j) = W_kc(i, j) + epsilon;
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
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);
            
            errMean = [];
            W_kc = temp;
            W_kc(i, j) = W_kc(i, j) - epsilon;
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
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);
            gW_kc_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    W_kc = temp;
    
    temp = W_in_H;
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            errMean = [];
            W_in_H = temp;
            W_in_H(i, j) = W_in_H(i, j) + epsilon;
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
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);
            
            errMean = [];
            W_in_H = temp;
            W_in_H(i, j) = W_in_H(i, j) - epsilon;
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
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);
            gW_in_H_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    
    % W_in_T
    temp = W_in_T;
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            errMean = [];
            W_in_T = temp;
            W_in_T(i, j) = W_in_T(i, j) + epsilon;
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
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);
            
            errMean = [];
            W_in_T = temp;
            W_in_T(i, j) = W_in_T(i, j) - epsilon;
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
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);
            gW_in_T_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    
    % W_R_T
    layer_idx = 3;
    temp = W_R_T{layer_idx};
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            errMean = [];
            W_R_T{layer_idx} = temp;
            W_R_T{layer_idx}(i, j) = W_R_T{layer_idx}(i, j) + epsilon;
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
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);
            
            errMean = [];
            W_R_T{layer_idx} = temp;
            W_R_T{layer_idx}(i, j) = W_R_T{layer_idx}(i, j) - epsilon;
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
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);
            gW_R_T_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    
    % W_R_H
    layer_idx = 2;
    temp = W_R_H{layer_idx};
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            errMean = [];
            W_R_H{layer_idx} = temp;
            W_R_H{layer_idx}(i, j) = W_R_H{layer_idx}(i, j) + epsilon;
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
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);
            
            errMean = [];
            W_R_H{layer_idx} = temp;
            W_R_H{layer_idx}(i, j) = W_R_H{layer_idx}(i, j) - epsilon;
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
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);
            gW_R_H_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    
    %% The gradient is godamn right ($$$$$$$$$$$   Keep Working here   $$$$$$$$$$$$$$$)
    fprintf('Check gW_R_H:%.10f\n', norm(gW_R_H_Check-gW_R_H));
    fprintf('Check gW_R_T:%.10f\n', norm(gW_R_T_Check-gW_R_T));
    fprintf('Check gW_in_H:%.10f\n', norm(gW_in_H_Check-gW_in_H));
    fprintf('Check gW_in_T:%.10f\n', norm(gW_in_T_Check-gW_in_T));
    fprintf('Check gW_kc:%.10f\n', norm(gW_kc_Check-gW_kc));
    % Gradient Check End
    
    % Update
    W_R_H = W_R_H - alpha * gW_R_H;
    W_R_T = W_R_T - alpha * gW_R_T;
    W_in_H = W_in_H - alpha * gW_in_H;
    W_in_T = W_in_T - alpha * gW_in_T;
    W_kc = W_kc - alpha * gW_kc;
    
    errArr = [errArr; mean(errMean)];
end
plotHandle = plot(errArr);


%% test
%% test
testLength = 10000;
pass = 0;
for tt = 1:testLength
    % Generate data
    [input, target, str] = embeded_reber_gen();
    x = input;
    d = target;
    succeed = 1;
    
    s = {};
    s{1} = zeros(n, 1);
    for t = 2:size(x, 1)
        % forward computation
        net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
        net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
        s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
        net_k = W_kc * s{t};
        y{t} = f_k(net_k);
        y_k = y{t};
        
        [sortedValues, sortedPos] = sort(y_k);
        pred = zeros(size(y_k));
        if str(t-1) == 'E'
            pred(sortedPos(end)) = 1;
        % There's only one target for the second symbol
        elseif (t-1)==2
            pred(sortedPos(end)) = 1;
        % Only one target for the last three symbols (e.g. ETE, EPE)
        elseif t>=(length(str)-2)
            pred(sortedPos(end)) = 1;
        else
            pred(sortedPos(end-1:end)) = 1;
        end
        tar = target(t-1, :);
        pred = pred';
        if sum(abs(tar - pred)) ~= 0 
            succeed = 0;
            break;
        end
    end
    if succeed==1
       pass = pass + 1; 
    end
end
pass