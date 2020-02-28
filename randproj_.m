function[projectionMatrix] = randproj_(k,d)

% randproj(k,d) returns a random projection matrix from d onto k dimensions.
% When d=k it returns a standard d-dimensional rotation or reflection
% matrix.

randNormMatrix = randn(k,d)*sqrt(1/d); % create a random matrix.

%randNormMatrix=randNormMatrix./repmat(sum(randNormMatrix.^2')',1,d);

%randNormMatrix(1,:)=randNormMatrix(1,:)*5;

projectionMatrix = randNormMatrix;
%projectionMatrix = orth(randNormMatrix')'; % orthogonalize the random matrix to generate the random projection matrix.


return

%changed temporarily!!!


projectionMatrix = randn(k,d)*sqrt(1/d); % create a random matrix.

[U,D,V]=svds(projectionMatrix);
projectionMatrix = randn(k,d)*sqrt(1/d); % create a random matrix.
[U1,D1,V1]=svds(projectionMatrix);

ind=randperm(k);

projectionMatrix = U(:,ind)*D1(ind,ind)*V(:,ind)';

