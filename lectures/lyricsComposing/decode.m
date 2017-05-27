function result = decode(input, vocabulary)
    [~, argIdx] = max(input, [], 2);
    for i =1:length(argIdx)
        key = num2str(argIdx(i)-1);
        if str2num(key)==250
            fprintf('\n');
        end
        fprintf('%s', vocabulary(key));
    end
    fprintf('\n');
end