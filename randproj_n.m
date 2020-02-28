function[projectionMatrix] = randproj_n(k,d)

% randproj(k,d) returns a random projection matrix from d onto k dimensions.
% When d=k it returns a standard d-dimensional rotation or reflection
% matrix.
randNormMatrix=zeros(k,d);
for i=1:k
    normVec=randn(1,d);
    normVec=normVec/norm(normVec);
randNormMatrix(i,:)=normVec;
end
projectionMatrix = randNormMatrix; % orthogonalize the random matrix to generate the random projection matrix.


