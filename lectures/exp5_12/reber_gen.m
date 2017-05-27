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