train = csvread('mnist_train.csv');
train_X = train(:, 2:end);
r = randi(size(train_X, 1), 10, 1);
centroid = train_X(r, :);
epoch = 50;
losses = zeros(epoch, 1);
for i = 1 : epoch
    [label, loss] = k_means(train_X, centroid);
    losses(i) = loss;
    for j = 1 : 10
        centroid(j, :) = mean(train_X(label == j - 1, :));
    end
end

plot(losses)