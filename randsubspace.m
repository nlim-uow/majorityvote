function[projectionMatrix] = randsubspace(k,d)

% randsubspace(k,d) returns a random subspace matrix from d onto k dimensions.
% When d=k it returns the identity matrix.

    temp=randperm(d,k);
    P=spalloc(k,d,k);
    %zeros(k,d);
    for i=1:size(temp,2) 
        P(i,temp(i))=1;
    end

projectionMatrix = P; % orthogonalize the random matrix to generate the random projection matrix.