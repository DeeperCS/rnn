clc; clear; close all;

f = @(x) 1 ./ (1 + exp(-x)) ;
df = @(x) f(x) .* (1 - f(x)) ;

W = -1 + 2 * rand(1, 1) ;

alpha = 0.2

P = zeros(1, 1) ;
S_t = zeros(1, 1) ;
S_t_1 = zeros(1, 1) ;

for i = 1:10000
    x = rand()<0.8 ;
    d = 1 - x ;

    net_t = W * S_t_1 + x ;

    S_t = S_t_1 + f(net_t) 

    P = P + df(net_t) * (S_t_1 + W * P);
%     P = P + df(net_t) * (S_t_1);
    
    S_t_1 = S_t;
    
    W = W - alpha * P * (S_t - d);
end

%% Test
S_t = zeros(1, 1) ;
S_t_1 = zeros(1, 1) ;

pass = 0 ;
for i = 1:10000
   x = rand()<0.8 ;

   d = 1 - x ;
   
   net_t = W * S_t_1 + x;
   S_t = S_t_1 + f(net_t);
   
   S_t_1 = S_t ;
   
   if (S_t>0.5) == d
       pass = pass + 1;
   end
end
pass