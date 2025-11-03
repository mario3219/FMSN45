function examinePrediction( data, predData, k, usedModel, noLags )

if nargin<5
    noLags = 50;
end

% Present normalized variance and MSE.
fprintf('Evaluating the prediction residual\n')
diffPred = data - predData;
varData  = var( data );
varPred  = var( diffPred );
if varPred > varData
    fprintf('  WARNING: prediction residual has higher variance than the data!\n');
end
fprintf('  The normalized variance of the prediction is %7.4f.\n', varPred/varData );

% If k==1, check if the residual is deemed white using the Monti test. 
if k==1
    checkIfWhite( diffPred );

    % Check if the used ACF is normal distributed
    pacfEst = pacf( diffPred, noLags, 0.05 );
    checkIfNormal( pacfEst(k+1:end), 'PACF' );
end

% Check for insignificant parameters
checkForSignificance( usedModel );

