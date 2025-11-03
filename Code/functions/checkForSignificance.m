% FUNCTION containsInSigParams = checkForSignificance( foundModel, [quietMode] )
%
% The function checks if the specified model contains statistically
% insignificant parameters. If quiteMode is unspecified it is by detault
% not quite and prints a statement if there are insignificant parameters.
%

% Reference: 
%   "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
%   Studentlitteratur, 2021
%
function containsInSigParams = checkForSignificance( foundModel, quietMode )

if nargin<2
    quietMode = 0;
end

estParams  = foundModel.Report.Parameters.ParVector( foundModel.Report.Parameters.Free );
paramStd   = sqrt(diag( foundModel.Report.Parameters.FreeParCovariance ));
testParams = abs(estParams) < paramStd; 
containsInSigParams = sum(testParams);

if containsInSigParams && ~quietMode
    fprintf('NOTE: The model contains parameters that are NOT statistically significant!\n');
end

