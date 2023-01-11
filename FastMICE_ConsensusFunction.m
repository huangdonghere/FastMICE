%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
% This is the source code for the consensus funtion in the FastMICE     %
% algorithm. If you find it helpful in your research, please cite the   %
% paper below.                                                          %
%                                                                       %
% Dong Huang, Chang-Dong Wang, Jian-Huang Lai.                          %
% Fast Multi-view Clustering via Ensembles: Towards Scalability,        %
% Superiority, and Simplicity.                                          %
% IEEE Transactions on Knowledge and Data Engineering, accepted, 2023.  %
%                                                                       %
% The code has been tested in Matlab R2019b on a PC with Windows 10.    %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function labels = FastMICE_ConsensusFunction(baseCls,k,maxTcutKmIters,cntTcutKmReps)
% Huang Dong. Mar. 20, 2019.
% Combine the M base clusterings in baseCls to obtain the final clustering
% result (with k clusters).

if nargin < 4
    cntTcutKmReps = 3; 
end
if nargin < 3
    maxTcutKmIters = 100; % maxTcutKmIters and cntTcutKmReps are used to limit the iterations of the k-means discretization in Tcut.
end

[N,M] = size(baseCls);

maxCls = max(baseCls);
for i = 1:numel(maxCls)-1
    maxCls(i+1) = maxCls(i+1)+maxCls(i);
end

cntCls = maxCls(end);
baseCls(:,2:end) = baseCls(:,2:end) + repmat(maxCls(1:end-1),N,1); clear maxCls

% Build the bipartite graph.
B=sparse(repmat([1:N]',1,M),baseCls(:),1,N,cntCls); clear baseCls
colB = sum(B);
B(:,colB==0) = [];

% Cut the bipartite graph.
labels = Tcut_for_bipartite_graph(B,k,maxTcutKmIters,cntTcutKmReps);
