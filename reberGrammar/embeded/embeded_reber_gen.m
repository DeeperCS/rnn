function [embeded_input, embeded_target, embeded_str] = embeded_reber_gen()
    %% Generate data using embeded reber grammar
    % Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural computation, 1997, 9(8): 1735-1780.
    % Minimal length is 9 for every sequence (comparative long time lags, and 3 nested dependencies)
    % This experiment demonstrates the RNN is able to act as a embeded FSM
    B = [1,0,0,0,0,0,0];
    T = [0,1,0,0,0,0,0];
    P = [0,0,1,0,0,0,0];
    E = [0,0,0,0,0,0,1];
    [embeded_input, embeded_target, embeded_str] = reber_gen();

    prob = rand();
    if prob>0.5
        embeded_str = ['T', embeded_str];
        embeded_input = [T; embeded_input];
        embeded_target = [B; embeded_target];
        
        embeded_str = [embeded_str, 'T'];
        embeded_input = [embeded_input; T];
        embeded_target(end, :) = T;  % modify target for the last symbol in reber sequence (it is not used in non embeded reber experiment)
        embeded_target = [embeded_target; E];
    else
        embeded_str = ['P', embeded_str];
        embeded_input = [P; embeded_input];
        embeded_target = [B; embeded_target];
        
        embeded_str = [embeded_str, 'P'];
        embeded_input = [embeded_input; P];
        embeded_target(end, :) = P; % modify target for the last symbol in reber sequence (it is not used in non embeded reber experiment)
        embeded_target = [embeded_target; E];
    end
    
    embeded_str = ['B', embeded_str];
    embeded_input = [B; embeded_input];
    embeded_target = [(T|P); embeded_target];
    
    embeded_str = [embeded_str, 'E'];
    embeded_input = [embeded_input; E];
    embeded_target = [embeded_target; E];
    
end

function [input, target, str] = reber_gen()
    %% input is the coded sequence
    %   target(1) is the target of input(1)
    B = [1,0,0,0,0,0,0];
    T = [0,1,0,0,0,0,0];
    P = [0,0,1,0,0,0,0];
    S = [0,0,0,1,0,0,0];
    X = [0,0,0,0,1,0,0];
    V = [0,0,0,0,0,1,0];
    E = [0,0,0,0,0,0,1];
    state = 0;
    str = [];
    input = [];
    target = [];
    while true
        prob = rand();
        if state == 0
            str = [str, 'B'];
            input = [input; B];
            state = 1;
            target = [target; (T|P)];        
        elseif state == 1
            if prob>0.5
                str = [str, 'T'];
                input = [input; T];
                state = 2;
                target = [target; (X|S)];

            else
                str = [str, 'P'];
                input = [input; P];
                state = 3;
                target = [target; (T|V)];
            end
        elseif state == 2
            if prob>0.5
                str = [str, 'S'];
                input = [input; S];
                state = 2;
                target = [target; (X|S)];
            else
                str = [str, 'X'];
                input = [input; X];
                state = 4;
                target = [target; (X|S)];
            end
        elseif state == 3
            if prob>0.5
                str = [str, 'T'];
                input = [input; T];
                state = 3;
                target = [target; (V|T)];
            else
                str = [str, 'V'];
                input = [input; V];
                state = 5;
                target = [target; (P|V)];
            end
        elseif state == 4
            if prob>0.5
                str = [str, 'S'];
                input = [input; S];
                state = 6;
                target = [target; E];
            else
                str = [str, 'X'];
                input = [input; X];
                state = 3;
                target = [target; (T|V)];
            end
        elseif state == 5
            if prob>0.5
                str = [str, 'V'];
                input = [input; V];
                state = 6;
                target = [target; (E)];
            else
                str = [str, 'P'];
                input = [input; P];
                state = 4;
                target = [target; (X|S)];
            end
        elseif state == 6
            str = [str, 'E'];
            input = [input; E];
            target = [target; E];
            break;
        end
    end
end