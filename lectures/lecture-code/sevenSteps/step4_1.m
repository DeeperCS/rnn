clc; clear all; close all;
f = @(x) 1 ./ (1 + exp(-x));
df = @(x) f(x) .* (1 - f(x));
h = f;
f_k = f;
dh = df;
df_k = df;

w_cu = -1 + 2 * rand(1, 2);
w_kc = -1 + 2 * rand(1, 1);

s_c(1) = 0;
p{1} = zeros(1, 2);
y_c(1) = 0;

alpha = 1;
T = 10000;
x = (rand(1, T) > 0.2) * 1;
d = [0 1-x];

netArr = [];
sArr = [];
pArr = [];
wArr = [];
yArr = [];
w_kc_Arr = [];
w_cc_Arr = [];
w_cx_Arr = [];

for t = 2:T
%     Feed Forward
    y_u{t-1} = [y_c(t-1); x(t-1)];
    net_c(t) = w_cu * y_u{t-1};
    s_c(t) = s_c(t-1) + f(net_c(t));
    y_c(t) = h(s_c(t));
    
    net_k(t) = w_kc * y_c(t);
    y_k(t) = f_k(net_k(t));
    
%     Update 
    delta_k(t) = ( y_k(t) - d(t) ) * df_k( net_k(t) );
    
    dw_kc = delta_k(t) * y_c(t);
    
    delta_c(t) = w_kc * delta_k(t) * dh(s_c(t));
    
    for u = 1:2
        p{t}(u) = p{t-1}(u) + df(net_c(t)) .* y_u{t-1}(u);
        dw_cu(u) = delta_c(t) * p{t}(u);
    end
    
    w_cu = w_cu - alpha * dw_cu;
    w_kc = w_kc - alpha * dw_kc;
    
    w_kc_Arr = [w_kc_Arr; w_kc];
    
    yArr = [yArr; y_k(t)];
%     netArr = [netArr; net(t)];
%     sArr = [sArr; s(t)];
%     pArr = [pArr; p(t)];
%     wArr = [wArr; w];
end

% plot(yArr, 'r');
plot(w_cx_Arr, 'r');
% plot(w_kc_Arr, 'r');
% plot(w_cc_Arr, 'r');
% hold on;
% plot(sArr, 'g');
% plot(pArr, 'b');
% plot(wArr, 'y');

s(1) = 0; % reset activation of the recurrent unit
T = 10000;
x = (rand(1, T) > 0.2) * 1;
d = [0 1 - x];
pass = 0;
for t = 2:T
    y_u{t-1} = [y_c(t-1); x(t-1)];
    net_c(t) = w_cu * y_u{t-1};
    s_c(t) = s_c(t-1) + f(net_c(t));
    y_c(t) = h(s_c(t));
    
    net_k(t) = w_kc * y_c(t);
    y_k(t) = f_k(net_k(t));
    
    if ((y_k(t) >= 0.5) - d(t)) == 0
        pass = pass + 1;
    end
end
pass