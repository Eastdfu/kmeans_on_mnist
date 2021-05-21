function[prediction] = NN(train_X, train_y, test_X)
prediction = zeros(size(test_X, 1), 1);
for i = 1 : size(test_X, 1)
    distance = zeros(size(train_X, 1), 1);
    for j = 1 : size(train_X, 1)
        distance(j) = dist(test_X(i, :), train_X(j, :));
    end
    [~, index] = min(distance);
    prediction(i) = train_y(index);
end
end


