%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
% This is the source code for the FastMICE algorithm, which is proposed %
% in the following paper:                                               %
%                                                                       %
% Dong Huang, Chang-Dong Wang, Jian-Huang Lai.                          %
% Fast Multi-view Clustering via Ensembles: Towards Scalability,        %
% Superiority, and Simplicity.                                          %
% IEEE Transactions on Knowledge and Data Engineering, accepted, 2023.  %
%                                                                       %
% The code has been tested in Matlab R2019b on a PC with Windows 10.    %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Label = runFastMICE(fea, trueK, distance)
% Huang Dong. Mar. 7, 2022.

if nargin < 3
    distance = 'euclidean';
end
% For text dataset, the Cosine distance is suggested.
% distance = 'Cosine'; 

lowerFeatureRatio = 0.2;   % Minimum ratio of feature sampling
upperFeatureRatio = 0.8;   % Maximum ratio of feature sampling

p = 1000;   % Total number of anchors in a view group
Knn = 5;    % Total number of nearest neighbors in a view group
Msize = 20; % Number of base clusterings

lowerK = 1; % The minimum number of clusters in a base clustering is lowerK * trueK
upperK = 2; % Maximum number of clusters in a base clustering

nView = length(fea);
N = size(fea{1},1);

% The number of anchors cannot exceed the number of instances.
if p > N
    p = N; 
end
    
%% Generate base clusterins
minK = lowerK*trueK;
maxK = upperK*trueK;
% tic1 = tic;
disp(['.']);
disp(['Start generating ', num2str(Msize), ' base clusterings via view-sharing bipartite graphs... ']);
disp(['.']);
IDX = FastMICE_EnsembleGeneration(fea, Msize, p, Knn, minK, maxK,lowerFeatureRatio,upperFeatureRatio,distance);
disp(['.']);
disp(['Ensemble generation completed.']);
% toc(tic1);

% tic2 = tic;
disp(['.']);
disp(['Start multi-view consensus function ... ']);
ticCon = tic;
Label = FastMICE_ConsensusFunction(IDX,trueK,20,5);
disp(['.']);
toc(ticCon);
disp(['Multi-view consensus functoin completed.']);
% toc(tic2);        
