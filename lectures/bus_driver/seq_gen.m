a = [1,0,0,0];
b = [0,1,0,0];
c = [0,0,1,0];
d = [0,0,0,1];

alphabet = [a;b;c;d];
str_alphabet = ['a','b','c','d'];
% isequal(alphabet(1,:), a)
x_seq = [];
t_seq = [];

first_a = 0;   
for i = 1:200
    idx = randi(4);
    if idx==2
        disp('a')
    end
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
    disp(followed_b)
end
