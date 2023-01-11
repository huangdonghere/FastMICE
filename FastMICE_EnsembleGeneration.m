%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
% This is the source code for ensemble generation in the FastMICE       %
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


function members = FastMICE_EnsembleGeneration(fea, M, p, Knn, lowK, upK,lowerFeatureRatio, upperFeatureRatio,distance)

if nargin < 9
    distance = 'euclidean';
end
if nargin < 8
    upperFeatureRatio = 0.8;
end
if nargin < 7
    lowerFeatureRatio = 0.2;
end
if nargin < 6
    upK = 60;
end
if nargin < 5
    lowK = 20;
end
if nargin < 4
    Knn = 5;
end
if nargin < 3
    p = 1000;
end
if nargin < 2
    M = 20;
end

nView = length(fea);
[N,~] = size(fea{1});
if p>N
    p = N;
end

rand('state',sum(100*clock)*rand(1)); % Reset the clock before generating random numbers
% For each of the M base clusterings, the number of clusters is randomly
% chosen in [lowK, upK].
Ks = randsample(lowK:upK,M,true);

% If the feature matrix is really sparse, it will be better to use sparse matrix.
for iV = 1:nView
    [~,d_features] = size(fea{iV});
    if sum(sum(fea{iV}~=0))/(N*d_features) < 0.02
        fea{iV} = sparse(fea{iV});
    end
end

warning('off');
% In ensemble generation, the iteration number in the kmeans discretization 
% of each base cluserer can be set to small values, so as to improve
% diversity of base clusterings and reduce the iteration time costs.
tcutKmIters = 5;
tcutKmRps = 1;

rand('state',sum(100*clock)*rand(1));
% For each of the M view groups, the number of view members is randomly
% chosen in [1, nView].
numViewsInGroups = randsample(1:nView,M,1);  
members = zeros(N,M);

rand('state',sum(100*clock)*rand(1));
% For each of the M base clusterings, a sampling ratio (for each view) is
% randomly selected.
sampleRatioAll = rand(M,nView)*(upperFeatureRatio-lowerFeatureRatio)+lowerFeatureRatio;

for i = 1:M
    % Generating the i-th base clustering.
    tic1 = tic;

    sampledFeatureIndex = [];
    for iV = 1:nView
       rand('state',sum(100*clock)*rand(1));
       sRatio = sampleRatioAll(i,iV);
       sampledFeatureIndex{iV} = randsample(1:size(fea{iV},2),ceil(sRatio*size(fea{iV},2)),0);
    end
    
    rand('state',sum(100*clock)*rand(1));
    % Randomly select $numViewsInGroups(i)$ views to form the i-th view
    % group.
    selectedViewIndex = randsample(1:nView,numViewsInGroups(i),0);
    members(:,i) = generateEachBaseClustering(fea, Ks(i), p, Knn, tcutKmIters, tcutKmRps, distance,selectedViewIndex,sampledFeatureIndex);

    toc(tic1);
end

function labels = generateEachBaseClustering(fea, Ks, sumP, sumKnn, maxTcutKmIters, cntTcutKmReps, distance,selectedViewIndex,sampledFeatureIndex)

N = size(fea{1},1);

eachP = ceil(sumP/numel(selectedViewIndex));
Knn = ceil(sumKnn/numel(selectedViewIndex));

if eachP>N
    eachP = N;
end

warning('off');

nView = length(fea);
B = [];
for iV = 1:nView
    if ~ismember(iV,selectedViewIndex)
        fea{iV} = [];
    else
        fea{iV} = fea{iV}(:,sampledFeatureIndex{iV});
    end
end
for iV = selectedViewIndex
    %% See our previous paper "Ultra-Scalable Spectral Clustering and 
    %% Ensemble Clustering" in IEEE TKDE 2020 to facilitate the understanding
    %% of this part.
    
    % Get $eachP$ representatives by hybrid selection for view $iV$
    RpFea = getRepresentativesByHybridSelection(fea{iV}, eachP,distance,10,5);
    if issparse(fea{iV})
        RpFea = sparse(RpFea);
    end
    
    %% Approx. KNN
    % 1. partition RpFea into $cntRepCls$ rep-clusters
    cntRepCls = floor(sqrt(eachP));
    % 2. find the center of each rep-cluster
    if strcmp(distance,'euclidean')
        [repClsLabel, repClsCenters] = litekmeans(RpFea,cntRepCls,'MaxIter',5);
    else
        [repClsLabel, repClsCenters] = litekmeans(RpFea,cntRepCls,'MaxIter',5,'Distance',distance);
    end
    
    % 3. Pre-compute the distance between N objects and the $cntRepCls$
    % rep-cluster centers
    if issparse(fea{iV})
        repClsCenters = sparse(repClsCenters);
    end
    if strcmp(distance, 'cosine')
        centerDist = pdist2_fast(fea{iV}, repClsCenters, distance);
    else
        centerDist = EuDist2(fea{iV}, repClsCenters,1);
    end
    
    %% Find the nearest rep-cluster (in RpFea) for each object
    [~,minCenterIdxs] = min(centerDist,[],2); clear centerDist
    cntRepCls = size(repClsCenters,1);
    %% Then find the nearest representative in the nearest rep-cluster for each object.
    nearestRepInRpFeaIdx = zeros(N,1);
    if strcmp(distance, 'cosine')
        for i = 1:cntRepCls
            [~,nearestRepInRpFeaIdx(minCenterIdxs==i)] = min(pdist2_fast(fea{iV}(minCenterIdxs==i,:),RpFea(repClsLabel==i,:), distance),[],2);
            tmp = find(repClsLabel==i);
            nearestRepInRpFeaIdx(minCenterIdxs==i) = tmp(nearestRepInRpFeaIdx(minCenterIdxs==i));
        end
    else
        for i = 1:cntRepCls
            [~,nearestRepInRpFeaIdx(minCenterIdxs==i)] = min(EuDist2(fea{iV}(minCenterIdxs==i,:),RpFea(repClsLabel==i,:), 1),[],2);
            tmp = find(repClsLabel==i);
            nearestRepInRpFeaIdx(minCenterIdxs==i) = tmp(nearestRepInRpFeaIdx(minCenterIdxs==i));
        end
    end
    clear repClsCenters repClsLabel minCenterIdxs tmp
    
    %% For each object, compute its distance to the candidate neighborhood of its nearest representative (in RpFea)
    neighSize = 10*Knn; % The candidate neighborhood size.
    if strcmp(distance, 'cosine')
        RpFeaW = pdist2_fast(RpFea,RpFea,distance);
    else
        RpFeaW = EuDist2(RpFea,RpFea,distance);
    end
    [~,RpFeaKnnIdx] = sort(RpFeaW,2); clear RpFeaW
    RpFeaKnnIdx = RpFeaKnnIdx(:,1:floor(neighSize+1)); % Pre-compute the candidate neighborhood for each representative.
    RpFeaKnnDist = zeros(N,size(RpFeaKnnIdx,2));
    if issparse(fea{iV})
        fea{iV} = full(fea{iV});
        RpFea = full(RpFea);
    end
    if strcmp(distance, 'cosine')
        for i = 1:eachP
            % fea{iV}(nearestRepInRpFeaIdx==i,:) denotes the objects (in fea{iV}) whose nearest representative is the i-th representative (in RpFea).
            RpFeaKnnDist(nearestRepInRpFeaIdx==i,:) = pdist2_fast(fea{iV}(nearestRepInRpFeaIdx==i,:), RpFea(RpFeaKnnIdx(i,:),:), distance);
        end
    else
        for i = 1:eachP
            % fea{iV}(nearestRepInRpFeaIdx==i,:) denotes the objects (in fea{iV}) whose nearest representative is the i-th representative (in RpFea).
            RpFeaKnnDist(nearestRepInRpFeaIdx==i,:) = EuDist2(fea{iV}(nearestRepInRpFeaIdx==i,:), RpFea(RpFeaKnnIdx(i,:),:), 1);
        end
    end
    fea{iV} = [];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear RpFea
    RpFeaKnnIdxFull = RpFeaKnnIdx(nearestRepInRpFeaIdx,:);
    
    %% Get the final KNN according to the candidate neighborhood.
    knnDist = zeros(N,Knn);
    knnTmpIdx = knnDist;
    knnIdx = knnDist;
    for i = 1:Knn
        [knnDist(:,i),knnTmpIdx(:,i)] = min(RpFeaKnnDist,[],2);
        temp = (knnTmpIdx(:,i)-1)*N+[1:N]';
        RpFeaKnnDist(temp) = 1e100;
        knnIdx(:,i) = RpFeaKnnIdxFull(temp);
    end
    clear RpFeaKnnIdx knnTmpIdx temp nearestRepInRpFeaIdx RpFeaKnnIdxFull RpFeaKnnDist
    
    %% Compute the cross-affinity matrix B for the bipartite graph
    if strcmp(distance,'cosine')
        Gsdx = 1-knnDist;
    else
        knnMeanDiff = mean(knnDist(:))+eps; % use the mean distance as the kernel parameter $\sigma$
        Gsdx = exp(-(knnDist.^2)/(2*knnMeanDiff^2)); clear knnMeanDiff
    end
    
    % normalize each row to unit norm
    Gsdx = bsxfun( @rdivide, Gsdx, sqrt(sum(Gsdx.*Gsdx,2))+eps );
    
    clear knnDist %%%%%%%%%
    Gsdx(Gsdx==0) = eps;
    Gidx = repmat([1:N]',1,Knn);
    B = [B,sparse(Gidx(:),knnIdx(:),Gsdx(:),N,eachP)]; clear Gsdx Gidx knnIdx
    % colSum = sum(B);
    % if sum(colSum(:)==0)>=1
    %     B(:,colSum==0) = []; % If a representative is not connected to any objects, then it will be removed.
    % end
end

clear fea
labels = zeros(N, numel(Ks));
for iK = 1:numel(Ks)
    labels(:,iK) = Tcut_for_bipartite_graph(B,Ks(iK),maxTcutKmIters,cntTcutKmReps,distance);
end


function RpFea = getRepresentativesByHybridSelection(fea, pSize, distance, cntTimes, repIters)
% Huang Dong. Mar. 20, 2019.
% Select $pSize$ representatives by hybrid selection.
% First, randomly select $pSize * cntTimes$ candidate representatives.
% Then, partition the candidates into $pSize$ clusters by k-means, and get
% the $pSize$ cluster centers as the final representatives.
if nargin < 5
    repIters = 20;
end
if nargin < 4
    cntTimes = 10;
end
if nargin < 3
    distance = 'euclidean';
end

N = size(fea,1);
bigPSize = cntTimes*pSize;
if pSize>N
    pSize = N;
end
if bigPSize>N
    bigPSize = N;
end

rand('state',sum(100*clock)*rand(1));
bigRpFea = getRepresentivesByRandomSelection(fea, bigPSize);

% [~, RpFea] = kmeans(bigRpFea,pSize,'MaxIter',20);
if strcmp(distance,'euclidean')
    [~, RpFea] = litekmeans(bigRpFea,pSize,'MaxIter',repIters);
else
    [~, RpFea] = litekmeans(bigRpFea,pSize,'MaxIter',repIters,'Distance',distance);
end


function [RpFea,selectIdxs] = getRepresentivesByRandomSelection(fea, pSize)
% Huang Dong. Mar. 20, 2019.
% Randomly select pSize rows from fea.

N = size(fea,1);
if pSize>N
    pSize = N;
end
selectIdxs = randperm(N,pSize);
RpFea = fea(selectIdxs,:);


function D = EuDist2(fea_a,fea_b,bSqrt)
%EUDIST2 Efficiently Compute the Euclidean Distance Matrix by Exploring the
%Matlab matrix operations.
%
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b
%
%    Examples:
%
%       a = rand(500,10);
%       b = rand(1000,10);
%
%       A = EuDist2(a); % A: 500*500
%       D = EuDist2(a,b); % D: 500*1000
%
%   version 2.1 --November/2011
%   version 2.0 --May/2009
%   version 1.0 --November/2005
%
%   Written by Deng Cai (dengcai AT gmail.com)


if ~exist('bSqrt','var')
    bSqrt = 1;
end

if (~exist('fea_b','var')) || isempty(fea_b)
    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    
    if issparse(aa)
        aa = full(aa);
    end
    
    D = bsxfun(@plus,aa,aa') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
    D = max(D,D');
else
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';

    if issparse(aa)
        aa = full(aa);
        bb = full(bb);
    end

    D = bsxfun(@plus,aa,bb') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
end