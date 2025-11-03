%
% Time series analysis
% Lund University
%
% Example exam code - please see the instructions below. You should only
% need to change this code in the places marked "CHANGE" when testing to
% ensure that your function works properly in the evaluation script. This
% code is only intended for you to test your code to ensure that it can run
% in a stand-alone manner correctly.   
% 
% Reference: 
%   "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
%   Studentlitteratur, 2021
%
clear; clc;
close all;
groupNo = 1;                                        % CHANGE. Specify your group number.
kStep   = 5;                                        % CHANGE. Set the prediction horizon.
noLags  = 200;                                      % CHANGE. Set the number of lags used in the ACF/PACF/etc.
addpath('../functions', '../data')


%% Set appropriate path and file structure.
% Note:
% * Your code should all be stored in a subfolder with the name "group001"
%   (if you are group one; note that the group numbering should have three
%   digits). Any functions you use should be stored directly in this folder
%   (not in subfolders). 
% * Your full code, report, etc can be in subfolders if you so wish. The
%   common code from the course book is stored in the folder "functions"
%   in the same directory as your folder (you do not need to include these
%   in your folder).   
grpStr = num2str( groupNo );
switch length(grpStr)
    case 1       
        grpStr = append("00", grpStr); 
    case 2       
        grpStr = append("0", grpStr); 
end
grpStr = append("group", grpStr);
addpath('../functions', grpStr)                     % Sets the path to your code.


%% Select data set
% All this code is for the example exam - you should load your data set
% here. This example corresponds to code28, where the predictor is derived.
% In this case, the output is stored as separate signals, so we create a
% data matrix for all relevant signals to submit to the predictors. If you
% are provided with a data matrix, this is the structure that should be
% submitted.
load dataHelsinki.mat                               % CHANGE.
outputdata   = powerload;                           % CHANGE.

dataMatrix(:,1) = outputdata;                       % Store the output data in the data matrix sent to the predictor.
% dataMatrix(:,2) = inputdata1;                     % Store the used input in the data matrix, with one column for each input.

testDataInd  = 7155:7826;                           % CHANGE. Select the test data. The exam code will use three different test periods.


%% Predict the the test data.  
% Note:
% * You should here make a call to your prediction functions.
% * You cannot give the functions any other input, the call must be as
%   indicated below.
% * Your functions should not estimate any parameters (using e.g. pem)
%   inside the function (other than using the Kalman filter of course). 
% * All parameters needed in the functions should be set (or loaded) inside
%   your function. These parameters are determined using the training data. 
% * Your functions should not open any figures or print any output on the
%   screen (except in case of errors). 
% * The function should be named with your group name (this function should
%   be directly inside your folder) and takes the entire data matrix as
%   input. There should be one function for each case: A, B, C, etc. 

% Form k-step predictions.
kStep = 3;
pred_ka = predCodeA_grp001( dataMatrix, testDataInd, kStep );       % CHANGE. Change to your group number.
%pred_kb = predCodeB_grp001( dataMatrix, testDataInd, kStep );      % CHANGE.
%pred_kc = predCodeC_grp001( dataMatrix, testDataInd, kStep );      % CHANGE.


%% Extract prediction residuals and compute prediction variances.
% Add results for other predictors as needed by copying the below.

% Form naive predictors.
[naive_k, var_naive_k] = naivePred(outputdata, testDataInd, kStep );%, seasonK );       % CHANGE? Should this have a season or not? Change accordingly. 

% Compute residuals. Below is just standard code to present the results.
varY = var( outputdata(testDataInd) );
ehatA_Pk = outputdata(testDataInd)-pred_ka;
%ehatB_Pk = outputdata(testDataInd)-pred_kb;
%ehatC_Pk = outputdata(testDataInd)-pred_kc;
var_predA_ka = var( ehatA_Pk );

% Show resulting prediction
figure
plot([outputdata(testDataInd) naive_k pred_ka ]),		
title(sprintf('%i-step prediction of the test data', kStep));
legend('Data', 'Naive', 'Proposed')
[~, pacfEst_ak] = plotACFnPACF( ehatA_Pk, noLags, sprintf('%i-step prediction using the proposed predictor', kStep));

fprintf('\nPrediction results for the %i-step prediction:\n', kStep)
fprintf('  The variance of the naive predictor is       %7.2f. The normalized variance is %5.4f.\n', var_naive_k, var_naive_k/varY )
fprintf('  The variance of the proposed A predictor is  %7.2f. The normalized variance is %5.4f.\n', var_predA_ka, var_predA_ka/varY )

if kStep==1
    checkIfWhite( ehatA_Pk );                                       % Only relevant if k=1.
    checkIfNormal( pacfEst_ak(kStep+1:end), 'PACF' );
end


