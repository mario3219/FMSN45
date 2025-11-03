%
% Time series analysis
% Lund University
%
% Example code 28: Creating a simple model for the Helsinki energy data and
% predicting it using the format of the examination code. See also the
% function examCode.m.
%
% Reference: 
%   "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
%   Studentlitteratur, 2021
%
clear; clc;
close all;
load dataHelsinki.mat
k       = 1;                                            % Prediction horizon.
groupNo = 1;                                            % Example group number.
noLags  = 200;                                          % Number of lags to consider.
sweek   = 168;                                          % Weekly season.
sday    = 24;                                           % Daily season.


%% Set appropriate path.
% Your exam code should all be stored in a subfolder with the name "group001"
% (if you are group one). Any functions you use should be stored in this
% folder. The common code from the course book is stored in the folder
% "functions" in the same directory as your folder (you do not need to 
% include these in your folder). 
grpStr = num2str( groupNo );
switch length(grpStr)
    case 1       
        grpStr = append("00", grpStr); 
    case 2       
        grpStr = append("0", grpStr); 
end
grpStr = append("group", grpStr);
addpath('../functions', '../data', grpStr)              % Sets the path to your code.


% Show entire data set, transform, and select training data.
figure 
plot(powerload),          
title('The data set: the energy consumption in Helsinki');

figure 
lambda_max = bcNormPlot(powerload,1)                    % Seems reasonable to use the sqrt as lambda = 0.4646
title('It seems reasonable to use a sqrt-transformation.')
data = sqrt( powerload );

trainDataInd = 5980:5980+sweek*6-1;                     % Select some training data; these 6 weeks seems nice.
trainingData = data(trainDataInd);                      % Notice that the model is build on only the training data.

figure, plot(trainingData),		
title('Transformed training data');


%% Model the data.
% Remove seasons and form an ARMA model. Don't forget to remove samples.
AS       = [1 -1];
dayPoly  = [1 zeros(1,sday-1) -1];
weekPoly = [1 zeros(1,sweek-1) -1];

y = filter(dayPoly,1,trainingData);     y = y(sday+1:end);
y = filter(weekPoly,1,y);               y = y(sweek+1:end);

% The model residual is not white, but reasonably close. We'll try this.
[armaModel, ey, ~, pacfEst] = estimateARMA( y, [ 1 1 1 ], conv(weekPoly, dayPoly), 'Model', 200 );


%% Form the k-step prediction using the examination format.
% Here, the Am and Cm vectors stored in the prediction function have been
% computed as:
%
%   Am = conv(armaModel.A, dayPoly);
%   Am = conv(Am, weekPoly);
%   Cm = armaModel.C;
%
% Notice that the prediction function does not take the parameters as
% input, so these need to be stored inside the function. Examine the call
% in the examCode.
testDataInd = 7155:7155+4*sweek-1;                      % This is the test data. You will not be told which data that is used to test the predictor.
yhatk = predCodeA_grp001( powerload, testDataInd, k );  % Notice that we call the function with data in the original domain; any needed transform has to be done inside the function.

figure
plot([powerload(testDataInd) yhatk] )
legend('Powerload','Prediction')
title(sprintf('%i-step prediction in the original domain', k))


%% Residual analysis
ehatA_P1 = powerload(testDataInd)-yhatk;
var_predA_1a = var( ehatA_P1 );
[~, pacfEst_a1] = plotACFnPACF( ehatA_P1, noLags, 'Prediction residual');

varY = var( powerload(testDataInd) );
fprintf('Prediction results for the one-step prediction:\n')
fprintf('  The variance of the validations data is   %7.2f\n', varY )
fprintf('  The variance of the proposed predictor is %7.2f. The normalized variance is %5.4f.\n', var_predA_1a, var_predA_1a/varY )
checkIfWhite( ehatA_P1 );
checkIfNormal( pacfEst_a1(k+1:end), 'PACF' );
