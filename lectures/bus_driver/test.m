a = [1,0,0,0];
b = [0,1,0,0];
c = [0,0,1,0];
d = [0,0,0,1];

alphabet = [a;b;c;d];
str_alphabet = ['a','b','c','d'];
x_seq = [];
t_seq = [];
testLength = 10000;

first_a = 0; 
for tt = 1:testLength
    % Generate data
    idx = randi(4);
    x = alphabet(idx, :);
    followed_b = 0;
    if isequal(x, a)
        first_a = 1;
    elseif isequal(x, b) & first_a == 1
        followed_b = 1;
        first_a = 0;
    end
    x_seq = [x_seq, str_alphabet(idx)];
    t_seq = [t_seq, followed_b];
    target = followed_b;
    
    % forward computation
    y_u = [y_c; x'];
    net_1 = w_1 * y_u;
    net_2 = w_2 * y_u;
    net_out = w_out * y_u;
    s_c = s_c + f1(net_1) .* f2(net_2);
    y_c = h(s_c) .* f_out(net_out);
    net_k = w_kc * y_c;
    y_k = f_k(net_k);

    err = 0.5 * (y_k - target)' * (y_k - target);
end