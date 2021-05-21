function [mean_vector, project_matrix] = PCA(X)
% feature指的是每一个像素点
% 求（每个feature的）平均值
mean_vector = mean(X);
% 把train set中心化
substracted_X = X - mean_vector;
% 求covariance_matrix， 代表了每个feature之间的关系
covariance_matrix = (substracted_X' * substracted_X) ./ (size(substracted_X, 1) - 1);
% 求covariance_matrix的特征值和特征矩阵
[vector, value] = eig(covariance_matrix);
% 把特征向量按特征值从大到小排序
[~, rank_idx] = sort(diag(value), 'descend');
project_matrix = vector(:, rank_idx);
end
