function[projectionMatrix] = randproj_3(k,d)

% randproj(k,d) returns a random projection matrix from d onto k dimensions.
% When d=k it returns a standard d-dimensional rotation or reflection
% matrix.
projectionMatrix = zeros(k,d);
sparsity=1/3;
d_1=ceil(sparsity * d / 2)*2;
for i=1:k
    randSelection = randperm(d,d_1); % create a random matrix.
    row = zeros(1,d);
    for j=1:(d_1 /2)
        row(randSelection(j))=sqrt(3);
    end
    for j=(d_1 /2)+1:d_1
        row(randSelection(j))=-sqrt(3);
    end
    projectionMatrix(i,:)=row;
end