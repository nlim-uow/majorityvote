function[projectionMatrix] = randproj_b(k,d)

% randproj(k,d) returns a random projection matrix from d onto k dimensions.
% When d=k it returns a standard d-dimensional rotation or reflection
% matrix.
projectionMatrix = double(rand(k,d)>0.5)*2-1;


