%% initialization
f = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df = @(x) f(x) .* (1 - f(x));

f_k = f;  df_k = df;
f_H = f; df_H = df;
f_T = f; df_T = df;
f_out = f;  df_out = df;

n = 400; % hidden
m = 2212; % input
k = 2212; % output
x = zeros(1, m);
x(1, randi(m)) = 1; % Random start
depth = 3;
s = {{}};
s{1}{depth} = zeros(n, 1);

errMean = [];
max_length = 200;
% generatedText = [];
% generatedText = [generatedText; x(1, :)];
for t = 2:max_length
    % forward computation
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
%     [~, argSort] = max(y{t}, []);
    choices = linspace(1, m, m);
    sampledIdx = randsample(choices, 1, true, y{t});
%     if sampledIdx==409
%        break 
%     end
    x_next = zeros(1, m);
    x_next(1, sampledIdx) = 1;
%     generatedText = [generatedText; x_next];
    x(t, :) = x_next;
end

decode(x, vocabulary)
% decode(generatedText, vocabulary)
