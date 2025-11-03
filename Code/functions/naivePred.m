% FUNCTION [naiveEst, var_naive, ehat] = naivePred(data, dataInd, kStep, [seasonK] )
%
% The function computes the naive kStep-predictor for data. If the
% optional parameter seasonK is given, the prediction is formed as the
% correspoding value last season, otherwise as the current value. The
% variance of the prediction residual is computed for the samples dataInd.
%

% Reference: 
%   "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
%   Studentlitteratur, 2021
%
function [naiveEst, var_naive, ehat] = naivePred(data, dataInd, kStep, seasonK )

% For the naive predictor. 
data = data(:);
naiveEst = zeros(size(data)); 
if nargin<4
    % Store the current value as the prediction at t+kStep.
    for t=1:length(data)-kStep
        naiveEst(t+kStep) = data(t);
    end

else
    % Instead use the corresponding value from last season and store the
    % value at t+kStep-seasonK as the prediction at t+kStep. 
    for t=1:length(data)-kStep
        if t-seasonK + kStep > 0 
            naiveEst(t+kStep) = data(t-seasonK + kStep );
        else
            naiveEst(t+kStep) = 0;
        end
    end
end
naiveEst = naiveEst(dataInd);

% Form prediction error and its variance.
ehat = data(dataInd)-naiveEst;
var_naive = var(ehat); 
