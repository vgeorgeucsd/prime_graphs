% Sample Spectral Clustering
% https://www.cs.purdue.edu/homes/dgleich/demos/matlab/spectral/spectral.html


clear all
close all
clc

n = 1000;

x = randperm(n);

gs = 450;

group1 = x(1:gs);
group2 = x(gs+1:end);

p_group1 = 0.5;
p_group2 = 0.4;
p_between = 0.1;

A(group1, group1) = rand(gs,gs) < p_group1;
A(group2, group2) = rand(n-gs,n-gs) < p_group2;
A(group1, group2) = rand(gs, n-gs) < p_between;

A = A .* ~eye(size(A));
A_directed = A;
% n = 1000
% k = 3
% A_directed = kregularHarary(n,k);

%% Undirected Graph Example
% A = triu(A,1);
% A = A + A';
% 
% Diagonals = eye(size(A));
% Diagonals(logical(eye(size(A)))) = sum(A);
% 
% L = Diagonals - A;
% graphTitle='Undirected Graph'
% getCommunities(L,graphTitle,A);

%% make prime graph
B = A_directed + speye(size(A_directed));

A_prime = [sparse(n,n) B; B' sparse(n,n)];

Diagonals_prime = eye(size(A_prime));
Diagonals_prime(logical(eye(size(A_prime)))) = sum(A_prime);

L_prime = Diagonals_prime - A_prime;

graphTitle='Prime Graph'
getCommunities(L_prime,graphTitle,A_prime);

%% Directed Graph Laplacian
DAG=A_directed;
[DAG_laplacian] = DirectedGraphLaplacian(DAG,'None');
graphTitle='Directed Graph'
getCommunities(DAG_laplacian,graphTitle,DAG);

% [DAG_laplacian2] = DirectedGraphLaplacian(DAG,0.90);
% graphTitle='Directed Laplacian 2'
% getCommunities(DAG_laplacian2,graphTitle,DAG);

%% Function to get the community structure
function getCommunities(L_prime,gType,G)
figure('Name',[gType, ' Adjacency Matrix']); 
spy(G);
title([gType ' Adjacency Matrix Before Sorting'])

[Vp,Dp, FLAGp] = eigs(L_prime,2,'SA');
Dp
FLAGp

figure('Name', [gType, ' Associated Eigenvector of 2nd Smallest Eig Val']); 
plot(Vp(:,2), '.-');
xlabel('Eigen Vector Dimension')
ylabel('Eigen Vector Values')
title([gType, ' Associated Eigenvector of 2nd Smallest Eig Val']);

figure('Name',[gType, ' Sorted Associated Eigenvector of 2nd Smallest Eig Val']); 
plot(sort(Vp(:,2)), '.-');
xlabel('Eigen Vector Dimension')
ylabel('Eigen Vector Values')
title([gType, ' Sorted Associated Eigenvector of 2nd Smallest Eig Val'])

[ignore, pp] = sort(Vp(:,2));

figure('Name',[gType, ' Sorted Index Values']); 
plot(pp);
title([gType ' Sorted Index Values'])


figure('Name',[gType, ' After Sorting']); 
spy(G(pp,pp));
title([gType ' Adjacency Matrix After Sorting'])
end