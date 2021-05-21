function [X] = apply_PCA(X, mean_vector, project_matrix, num_dimensions)
X = X - mean_vector;
X = X * project_matrix(:, 1 : num_dimensions);
end