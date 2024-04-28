function [DAG_laplacian,P] = DirectedGraphLaplacian(DAG_Original, Alpha)
%% Directed Graph Laplacian
% This function returns the directed Laplacian matrix of any graph (DAG).
% This is calculated using the following formula
% L = I - (Phi^{1/2} * P  * Phi^{-1/2} + Phi^{-1/2} * P^T * Phi^{1/2} ) / 2
% where,
%     I   : Identity matrix, 
%     Phi : Matrix with the Perron vector of P in the diagonal and zeros elsewhere, and 
%     P   : Transition matrix of the graph. This value depends on the walk
%           type of the graph exploration. 
%           !! Current implementation includes only a "PageRank" walk type. 
%           !! Future Implementation Plan: "Random Walk"
% Referred paper:
% Fan Chung (2005). Laplacians and the Cheeger inequality for directed graphs.Annals of Combinatorics, 9(1), 2005

%% Function Parametres
% Input : 
%           DAG             : Adjacency Matrix of a Directed Network.
%           Alpha           : Confidence Interval Value for PageRank 
%                             Takes either 'None' or User Defined Value
% Output: 
%           DAG_Laplacian   : Directed Laplacian Matrix of the graph.

%% DAG from original DAG input
DAG = DAG_Original;
%% Initial Parameters
% If "Alpha" value is defined then use it or defaul value is set to ".95"
if Alpha == 'None'
    alpha=0.95;     % Confidence Level
else
    alpha = Alpha
end
% DAG Size
[row,~]=size(DAG);

%% Calculating the Parameters
%% Adding a pertebrated value for all zerow rows to avoide error in calculation
% Finding Rows with zeros
row_has_all_zeros = ~any(DAG,2);
% Getting the indices of those zeros
Indices = find(row_has_all_zeros);
% Finding Rows with zeros and add a constant
DAG(Indices,:) = eps; %1.0 / row; % Normalized Row

% Normalize the whole Matrix
Normalized_DAG_Sum=sum(DAG,2);

for i=1:row
    DAG(i,:)=DAG(i,:)/Normalized_DAG_Sum(i);
end

%% "P" Matrix
P = alpha * DAG + (1 - alpha) / row;

%% Get the Eigen Value
[~,~,W]=eig(P);
EigenVector=real(W(:,1));

p = EigenVector / sum(EigenVector);

%% Calculate "phi"
SquareRoot_P = sqrt(p);
SquareRoot_P_Inverse=1./SquareRoot_P;

%% Z matrix
Z = spdiags(SquareRoot_P,0,row,row) * P * spdiags(SquareRoot_P_Inverse,0,row,row);

%% Identity Matrix
I = eye(size(DAG,2));

%% Calculate Directed Graph laplacian
DAG_laplacian = I - (Z + Z') / 2.0 ; % Formula as referred in the paper

end

