clc; clear all; close all;
f = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df = @(x) f(x) .* (1 - f(x));
h = @(x) 2 .* f(x) - 1; % 2 * sigmoid - 1
dh = @(x) 2 .* df(x);
f_k = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df_k = @(x) f_k(x) .* (1 - f_k(x));
g = @(x) 1 ./ (1 + exp(-x)); % sigmoid
dg = @(x) f_k(x) .* (1 - f_k(x));
f_in = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df_in = @(x) f(x) .* (1 - f(x));

%% initialization
w_cu = -1 + 2 * rand(1, 2);
w_kc = -1 + 2 * rand(1, 1);
w_in = -1 + 2 * rand(1, 1);
s_c(1) = 0;
y_c(1) = 0;
p_u{1} = zeros(1, 2);
p_in{1} = 0;

%% train
alpha = 0.2;
T = 1000;
x = (rand(1, T) > 0.2) * 1;
d = [0 1-x];
for t = 2:T
    % forward computation
    y_u{t-1} = [y_c(t-1);x(t-1)];
    net_c(t) = w_cu * y_u{t-1}; % net^c(t) = \sum_{u}{w_{cu} y_{u}(t-1)}
    net_in(t) = w_in * y_c(t-1);
    y_in(t) = f_in(net_in(t));
    s_c(t) = s_c(t-1) + y_in(t) * g(net_c(t));
    
    y_c(t) = h(s_c(t));
    net_k(t) = w_kc * y_c(t);
    y_k(t) = f_k(net_k(t));
    
    % compute the gradient of w_kc
    dK(t) = (y_k(t) - d(t)) * df_k(net_k(t));
    gW_kc = dK(t) * y_c(t);
    % update gradient information
    dSc(t) = dK(t) * w_kc * dh(s_c(t));
    p_in{t} = p_in{t-1} + df_in(net_in(t)) * g(net_c(t)) * y_c(t-1);
    gW_in = dSc(t) * p_in{t};
    for u = 1:2
        p_u{t}(u) = p_u{t-1}(u) + f_in(net_in(t)) * dg(net_c(t)) * y_u{t-1}(u);
        gW_cu(u) = dSc(t) * p_u{t}(u);
    end
    % update weights
    w_in = w_in - alpha * gW_in;
    w_kc = w_kc - alpha * gW_kc;
    w_cu = w_cu - alpha * gW_cu;
end

%% test
s_c(1) = 0; % reset activation of the recurrent unit
y_c(1) = 0;
T = 10000;
x = (rand(1, T) > 0.2) * 1;
d = [0 1-x];
pass = 0;
for t = 2:T
    y_u{t-1} = [y_c(t-1);x(t-1)];
    net_c(t) = w_cu * y_u{t-1}; % net^c(t) = \sum_{u}{w_{cu} y_{u}(t-1)}
    net_in(t) = w_in * y_c(t-1);
    y_in(t) = f_in(net_in(t));
    s_c(t) = s_c(t-1) + y_in(t) * g(net_c(t));
    
    y_c(t) = h(s_c(t));
    net_k(t) = w_kc * y_c(t);
    y_k(t) = f_k(net_k(t));
    
    if ((y_k(t) >= 0.5) - d(t)) == 0
        pass = pass + 1;
    end
end
pass