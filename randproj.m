function[projectionMatrix] = randproj(k,d)

% randproj(k,d) returns a random projection matrix from d onto k dimensions.
% When d=k it returns a standard d-dimensional rotation or reflection
% matrix.

randNormMatrix = randn(k,d); % create a random matrix.
projectionMatrix = orth(randNormMatrix')'; % orthogonalize the random matrix to generate the random projection matrix.


