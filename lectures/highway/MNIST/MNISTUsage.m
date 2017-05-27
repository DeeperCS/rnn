images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
[width height] = size(images(1, 1:100))
display_network(images(:, 200:400));
disp(labels(1:10));