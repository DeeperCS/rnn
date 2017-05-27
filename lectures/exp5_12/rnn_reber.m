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
n = 4; % hidden
m = 7; % input
k = 7; % output

W_I = -1 + 2 * rand(n, m);
W_R = -1 + 2 * rand(n, n);
W_kc = -1 + 2 * rand(k, n);

gW_I = zeros(size(W_I));
gW_R = zeros(size(W_R));
gW_kc = zeros(size(W_kc));

%% train
alpha = 0.5;

trainLength = 1;
errArr = [];
% plotHandle = plot(errArr);
for tt = 1:trainLength
    % Generate data
    [input, target, str] = embeded_reber_gen();
    x = input;
    T = size(x, 1)
    s{1} = zeros(n, 1);
    net{1} = zeros(n, 1);
    % Accumulate gradients for every time step    
    gW_I = zeros(size(W_I));
    gW_R = zeros(size(W_R));
    gW_kc = zeros(size(W_kc));
    
    errMean = [];
    for t = 2:size(x, 1)
        % forward computation
        d = x(t, :)';
        net{t} = W_I * x(t-1, :)' + W_R * s{t-1};
        s{t} = f(net{t});

        % Using the last layer as output
        net_k{t} = W_kc * s{t};
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
        gW_kc = gW_kc + delta_K * s{t}';
        
        % backpropagation to s
        delta_s{t} = (W_kc' * delta_K) + delta_next;
        
        gW_R = gW_R + delta_s{t} .* df(net{t}) * s{t-1}';
        gW_I = gW_I + delta_s{t} .* df(net{t}) * x(t-1, :);
        
        delta_s{t-1} = delta_s{t} .* df(net{t});
      
        delta_next = delta_s{t-1};
    end
    
    
    %% Gradient check
%     temp = W_kc;
%     gW_Check = zeros(size(temp));
%     epsilon = 0.000001;
%     for i = 1:size(temp, 1)
%         for j = 1:size(temp, 2)
%             W_kc = temp;
%             W_kc(i, j) = W_kc(i, j) + epsilon;
%             % Feedforward
%             x = input;
%             T = size(x, 1);
%             s{1} = zeros(n, 1);
%             net{1} = zeros(n, 1);
% 
%             errMean = [];
%             for t = 2:size(x, 1)
%                 % forward computation
%                 d = x(t, :)';
%                 net{t} = W_I * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R * [s{t-1}; ones([size(s{t-1}, 2), 1])];
%                 s{t} = s{t-1} + ( f(net{t}) - s{t-1} );
% 
%                 % Using the last layer as output
%                 net_k{t} = W_kc * [s{t}; ones([size(s{t}, 2), 1])];
%                 y{t} = f_k(net_k{t});
%                 y_k = y{t};
%                 err = 0.5 * (y_k - d)' * (y_k - d);
%                 errMean = [errMean; err];
%             end
%             E1 = sum(errMean);
% 
%             W_kc = temp;
%             W_kc(i, j) = W_kc(i, j) - epsilon;
%             % Feedforward
%             x = input;
%             T = size(x, 1);
%             s{1} = zeros(n, 1);
%             net{1} = zeros(n, 1);
% 
%             errMean = [];
%             for t = 2:size(x, 1)
%                 % forward computation
%                 d = x(t, :)';
%                 net{t} = W_I * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R * [s{t-1}; ones([size(s{t-1}, 2), 1])];
%                 s{t} = s{t-1} + ( f(net{t}) - s{t-1} );
% 
%                 % Using the last layer as output
%                 net_k{t} = W_kc * [s{t}; ones([size(s{t}, 2), 1])];
%                 y{t} = f_k(net_k{t});
%                 y_k = y{t};
%                 err = 0.5 * (y_k - d)' * (y_k - d);
%                 errMean = [errMean; err];
%             end
%             E2 = sum(errMean);
% 
%             gW_Check(i, j) = (E1 - E2) / (2*epsilon);
%         end
%     end
%     disp('gW_kc')
%     gW_Check
    
    %% Gradient check R
    temp = W_R;
    gW_Check = zeros(size(temp));
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            W_R = temp;
            W_R(i, j) = W_R(i, j) + epsilon;
            % Feedforward
            x = input;
            T = size(x, 1);
            s{1} = zeros(n, 1);
            net{1} = zeros(n, 1);

            errMean = [];
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net{t} = W_I * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R * [s{t-1}; ones([size(s{t-1}, 2), 1])];
                s{t} = f(net{t});

                % Using the last layer as output
                net_k{t} = W_kc * [s{t}; ones([size(s{t}, 2), 1])];
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);

            W_R = temp;
            W_R(i, j) = W_R(i, j) - epsilon;
            % Feedforward
            x = input;
            T = size(x, 1);
            s{1} = zeros(n, 1);
            net{1} = zeros(n, 1);

            errMean = [];
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net{t} = W_I * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R * [s{t-1}; ones([size(s{t-1}, 2), 1])];
                s{t} = s{t-1} + ( f(net{t}) - s{t-1} );

                % Using the last layer as output
                net_k{t} = W_kc * [s{t}; ones([size(s{t}, 2), 1])];
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);

            gW_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    disp('gW_R')
    gW_Check
    
    %% Gradient check I
    temp = W_I;
    gW_Check = zeros(size(temp));
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            W_I = temp;
            W_I(i, j) = W_I(i, j) + epsilon;
            % Feedforward
            x = input;
            T = size(x, 1);
            s{1} = zeros(n, 1);
            net{1} = zeros(n, 1);

            errMean = [];
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net{t} = W_I * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R * [s{t-1}; ones([size(s{t-1}, 2), 1])];
                s{t} = s{t-1} + ( f(net{t}) - s{t-1} );

                % Using the last layer as output
                net_k{t} = W_kc * [s{t}; ones([size(s{t}, 2), 1])];
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);

            W_I = temp;
            W_I(i, j) = W_I(i, j) - epsilon;
            % Feedforward
            x = input;
            T = size(x, 1);
            s{1} = zeros(n, 1);
            net{1} = zeros(n, 1);

            errMean = [];
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net{t} = W_I * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R * [s{t-1}; ones([size(s{t-1}, 2), 1])];
                s{t} = s{t-1} + ( f(net{t}) - s{t-1} );

                % Using the last layer as output
                net_k{t} = W_kc * [s{t}; ones([size(s{t}, 2), 1])];
                y{t} = f_k(net_k{t});
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);

            gW_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    disp('gW_I')
    gW_Check
    
    
    % Update
    W_I = W_I - alpha * gW_I;
    W_R = W_R - alpha * gW_R;
    W_kc = W_kc - alpha * gW_kc;

    errArr = [errArr; sum(errMean)];
end
plotHandle = plot(errArr);



    
    
    
    
    
    


%% test
testLength = 10000;
pass = 0;
for tt = 1:testLength
    % Generate data
    [input, target, str] = embeded_reber_gen();
    x = input;
    d = target;
    succeed = 1;
    
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