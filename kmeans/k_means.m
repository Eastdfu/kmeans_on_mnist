function[label, loss] = k_means(X, centroid)
label = zeros(size(X, 1), 1);
loss = 0;
for i = 1 : size(X, 1)
    distance = zeros(10, 1);
    for j = 1 : 10
        distance(j) = dist(X(i, :), centroid(j, :));
    end
    [d, index] = min(distance);
    loss = loss + d;
    label(i) = index - 1;
end
end