clc; clear all; close all;
f = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df = @(x) f(x) .* (1 - f(x));

h = @(x) 1 ./ (1 + exp(-x)); % sigmoid
dh = @(x) f(x) .* (1 - f(x));

f_k = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df_k = @(x) f_k(x) .* (1 - f_k(x));

f1 = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df1 = @(x) f1(x) .* (1 - f1(x));

f2 = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df2 = @(x) f2(x) .* (1 - f2(x));

f_out = @(x) 1 ./ (1 + exp(-x)); 
df_out = @(x) f_out(x) .* (1 - f_out(x));

%% initialization
w_1 = -1 + 2 * rand(1, 3);
w_2 = -1 + 2 * rand(1, 3);
w_out = -1 + 2 * rand(1, 3);
w_kc = -1 + 2 * rand(1, 1);

s_c(1) = 0;
y_c(1) = 0;
p1_u{1} = zeros(1, 3);
p2_u{1} = zeros(1, 3);

%% train
alpha = 0.2;
T = 1000;
x = (rand(2, T) > 0.5) * 1;
d = [0, xor(x(1, :), x(2, :))];

for t = 2:T
    % forward computation
    y_u{t-1} = [y_c(t-1); x(:, t-1)];
    
    net_1(t) = w_1 * y_u{t-1};
    net_2(t) = w_2 * y_u{t-1};
    net_out(t) = w_out * y_u{t-1};
        
    y_1(t) = f1(net_1(t));
    y_2(t) = f2(net_2(t));
    y_out(t) = f_out(net_out(t));
    
    s_c(t) = s_c(t-1) + y_1(t) * y_2(t);
    
    y_c(t) = h(s_c(t)) * y_out(t);
    
    net_k(t) = w_kc * y_c(t);
    y_k(t) = f_k(net_k(t));
    
    % compute the gradient of w_kc
    dK(t) = (y_k(t) - d(t)) * df_k(net_k(t));
    gW_kc = dK(t) * y_c(t);
    % update gradient information
    dSc(t) = dK(t) * w_kc * y_out(t) * dh(s_c(t));
    dOut(t) = dK(t) * w_kc * df_out(net_out(t)) * h(s_c(t));
    for u = 1:3
        p1_u{t}(u) = p1_u{t-1}(u) + df1(net_1(t)) * f2(net_2(t)) * y_u{t-1}(u);
        gW1(u) = dSc(t) * p1_u{t}(u);
        
        p2_u{t}(u) = p2_u{t-1}(u) + f1(net_1(t)) * df2(net_2(t)) * y_u{t-1}(u);
        gW2(u) = dSc(t) * p2_u{t}(u);
        
        gWout(u) = dOut(t) * y_u{t-1}(u);
    end  
    
    % update weights
    w_1 = w_1 - alpha * gW1;
    w_2 = w_2 - alpha * gW2;
    w_kc = w_kc - alpha * gW_kc;
end

%% test
s_c(1) = 0; % reset activation of the recurrent unit
y_c(1) = 0;
T = 10000;
x = (rand(2, T) > 0.5) * 1;
d = [0, xor(x(1, :), x(2, :))];
pass = 0;
for t = 2:T
    y_u{t-1} = [y_c(t-1); x(:, t-1)];
    
    net_1(t) = w_1 * y_u{t-1};
    net_2(t) = w_2 * y_u{t-1};
    net_out(t) = w_out * y_u{t-1};
        
    y_1(t) = f1(net_1(t));
    y_2(t) = f2(net_2(t));
    y_out(t) = f_out(net_out(t));
    
    s_c(t) = s_c(t-1) + y_1(t) * y_2(t);
    
    y_c(t) = h(s_c(t)) * y_out(t);
    
    net_k(t) = w_kc * y_c(t);
    y_k(t) = f_k(net_k(t));
    
    if ((y_k(t) >= 0.5) - d(t)) == 0
        pass = pass + 1;
    end
end
pass