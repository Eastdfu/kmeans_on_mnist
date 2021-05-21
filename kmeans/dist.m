function[d] = dist(x, y)
temp = x - y;
temp = temp.^2;
d = sqrt(sum(temp));
end