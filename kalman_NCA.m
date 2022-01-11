function [X,P] = kalman_NCA(F,GQGt,P,H,R,Z,X)    
% X: state vector
% P: covariance matrix
% F: kinematic motion constrained between t(k) and t(k+1)
% H: output matrix that relates the kinematic state to the measurement (observation m44444444444444666666666666444444444444444444444444444atrix)
% ¿?R: covariance of the sensor measurement error 
% ¿?Z: sensor measurement vector
% ¿w?: sensor error (normal scalar variable with variance sigmaw^2)
% El primer valor no nulo de X será K·Z
%% prediction
    X = F * X;
    P = F * P * F' + GQGt;
%% update
   % Z = H * X + w;
    S = H * P * H' + R;
    K = P * H' * inv(S);
    X = X + K * (Z - H * X);
    P = (eye(3) - K * H) * P;
    
    