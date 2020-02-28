function[matrix] = randnormalized(k,d)

% randproj(k,d) returns a random projection matrix from d onto k dimensions.
% When d=k it returns a standard d-dimensional rotation or reflection
% matrix.

matrix = randn(k,d); % create a random matrix.
for i=1:k
matrix(i,:)=matrix(i,:)/norm(matrix(i,:));    
end



