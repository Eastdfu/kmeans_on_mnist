train = csvread('mnist_train.csv');
train_y = train(:, 1);
train_X = train(:, 2:end);
[Mu, P] = PCA(train_X);

reduced_train_X = apply_PCA(train_X, Mu, P, 256);
evaluations = zeros(246, 1);

for d = 256 : -1 : 10
    temp = reduced_train_X(:, 1 : d);
    centroid = zeros(10, size(temp, 2));
    % 初始化的点影响了后面预测的正确率 这是目前随机出来的点 也可以自己随机一下初始来提高正确率
    centroid(1, :) = temp(667, :);
    centroid(2, :) = temp(5430, :);
    centroid(3, :) = temp(3570, :);
    centroid(4, :) = temp(1109, :);
    centroid(5, :) = temp(2453, :);
    centroid(6, :) = temp(514, :);
    centroid(7, :) = temp(2546, :);
    centroid(8, :) = temp(3618, :);
    centroid(9, :) = temp(1781, :);
    centroid(10, :) = temp(5879, :);

    epoch = 50;
    acc = zeros(10, 1);
    losses = zeros(epoch, 1);

    for i = 1 : epoch
        [label, loss] = k_means(temp, centroid);
        losses(i) = loss;
        for j = 1 : 10
            centroid(j, :) = mean(temp(label == j - 1, :));
        end
    end

    for i = 1 : 10
        acc(i) = sum(label(train_y == i - 1) == i - 1) / sum(train_y == i - 1);
    end
    
    evaluations(256 - d + 1) = mean(acc);  
end

x = 256 : -1 : 10;
plot(x, evaluations);