function [data_train, label_train, data_test, label_test] = load_cifar10()
    data_train = [];
    label_train = [];
    data_test = [];
    label_test = [];
    
    for i = 1:5
        filename = ['data_batch_', num2str(i)];
        load(filename);
        data_train = [data_train; data];
        label_train = [label_train; labels];
    end
    
    filename = 'test_batch';
    load(filename);
    data_test = [data_test; data];
    label_test = [label_test; labels];
end