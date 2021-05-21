train = csvread('mnist_train.csv');
test = csvread('mnist_test.csv');
train_y = train(:, 1);
train_X = train(:, 2:end);
test_y = test(:, 1);
test_X = test(:, 2:end);
[Mu, P] = PCA(train_X);
reuduced_train_X = apply_PCA(train_X, Mu, P, 256);
reuduced_test_X = apply_PCA(test_X, Mu, P, 256);

errors = zeros(246, 1);
for i = 256 : -1 : 10
    prediction = NN(reuduced_train_X(:, 1 : i), train_y, reuduced_test_X(:, 1 : i));
    errors(256 - i + 1) = sum(prediction ~= test_y) / 1000;
end
x = 256 : -1 : 10;
plot(x, errors);