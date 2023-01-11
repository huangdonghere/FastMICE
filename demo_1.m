%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                      %
% This is a demo for the FastMICE algorithm, which is proposed in the  %
% following paper:                                                     %
%                                                                      %
% Dong Huang, Chang-Dong Wang, Jian-Huang Lai.                         %
% Fast Multi-view Clustering via Ensembles: Towards Scalability,       %
% Superiority, and Simplicity.                                         %
% IEEE Transactions on Knowledge and Data Engineering, accepted, 2023. %
%                                                                      %
% The code has been tested in Matlab R2019b on a PC with Windows 10.   %
%                                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demo_1()
%% Run the FastMICE algorithm multiple times and show its average performance.

clear all;
close all;
clc;


%% Load the data.
% Please uncomment the dataset that you want to test.
% dataName = 'OutScene';
dataName = 'ALOI';

load(['data_',dataName,'.mat'],'fea','gt'); 
V = length(fea);
N = numel(gt);
K = numel(unique(gt)); % The number of clusters

% The number of times that the FastMICE algorithm will be run.
cntTimes = 5; 
% Save the scores of NMI, ARI, ACC, and PUR.
fourScores = zeros(cntTimes,4);

%% Run FastMICE
for runIdx = 1:cntTimes
    disp('**************************************************************');
    disp(['Run ', num2str(runIdx),':']);
    disp('**************************************************************');
    
    Label = runFastMICE(fea, K);  
    
    fourScores(runIdx,:) = computeFourClusteringMetrics(Label,gt);
    
    disp('--------------------------------------------------------------'); 
    disp(['Run ',num2str(runIdx), ': ']);   
    disp(['NMI = ',num2str(fourScores(runIdx,1))]);
    disp(['ARI = ',num2str(fourScores(runIdx,1))]);
    disp(['ACC = ',num2str(fourScores(runIdx,1))]);
    disp(['PUR = ',num2str(fourScores(runIdx,1))]);
    disp('--------------------------------------------------------------');
end

disp('**************************************************************');
disp(['  ** Average Performance over ',num2str(cntTimes),' runs on the ',dataName,' dataset **']);
disp('--------------------------------------------------------------');
disp(['Sample size: N = ', num2str(N)]);
disp(['Average NMI score: ',num2str(mean(fourScores(:,1)))]);
disp(['Average ARI score: ',num2str(mean(fourScores(:,2)))]);
disp(['Average ACC score: ',num2str(mean(fourScores(:,3)))]);
disp(['Average PUR score: ',num2str(mean(fourScores(:,4)))]);
disp('--------------------------------------------------------------');
disp('**************************************************************');