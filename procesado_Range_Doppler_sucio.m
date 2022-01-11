
    function [Mrd, eje_R, eje_vel] = procesado_Range_Doppler_sucio(srx, t , senal, system, procesador)    
        %**********************************************************************
    % 1º REORDENAR VECTOR SRX EN MATRIZ. CADA FILA UNA DE LAS "NPULSOS" RAMPAS
    % ver "reshape.m"
    %
    % 2º REALIZAR LA MEZCLA CON LA SEÑAL DE REFERENCIA TRANSMITIDA. SERÍA MULTIPLICAR 
    % CADA FILA (RAMPA) POR LA CONJUGADA DE LA SEÑAL IDEAL DE UN BLANCO A DISTANCIA
    % 0 METROS. sref = conj(exp(1i*pi*senal.gamma.*(t.^2))).
    %
    % 3ºCOMPRESIÓN EN DISTANCIA. REALIZAR FFT POR FILAS (EN DISTANCIA), PREVIA MULTIPLICACIÓN DE CADA FILA POR VENTANA DE
    % TAYLOR "taylorwin.m", "fft.m"
    %
    % 4ºCOMPRESIÓN EN DOPPLER. REALIZAR FFT POR COLUMNAS (EN DOPPLER), PREVIA MULTIPLICACIÓN DE CADA COLUMNA POR
    % VENTANA DE TAYLOR "taylorwin.m", "fft.m"
    %
    % 5º CALCULAR LOS EJES DISTANCIA/VELOCIDAD DE LA MATRIZ RANGE/DOPPLER
    % "imagesc.m"
    %*********************************************************************
    
    %% Secuencia de señales transmitidas (demodulación chirp por chirp)
% 
%db = 0;
% 
% stx = 0;
% 
% for np = 1:senal.npulsos,    
% 
%         tp0 = (np-1)*senal.PRI;      
% 
%         pulsor = rectpuls((t-(tp0+senal.tau/2+db))/senal.tau);
% 
%         pulsor = pulsor.*exp(1i*2*pi*(tp0+senal.tau/2+db)*senal.f0).*exp(1i*pi*senal.gamma.*(t-(tp0+senal.tau/2+db)).^2);
% 
%                
% 
%         stx = stx + pulsor;
% 
% end
% 
% sref = conj(stx);
% 
% 
% demod_Mrd = reshape(sref.*srx, [senal.PRI*system.fs, senal.npulsos]).';   % Demodulación y formateo simultáneos
   

%% Demodulación por fila (rampa limitada)

shaped_Mrd = reshape(srx, [senal.PRI*system.fs, senal.npulsos]).';

taux=t(1: senal.PRI*system.fs);

sref = conj(exp(1i*pi*senal.gamma.*((taux - senal.tau/2).^2)));

demod_Mrd = bsxfun(@times, shaped_Mrd, sref);


    %% Enventanados simultáneos
    
%     w_r = taylorwin(senal.PRI*system.fs,4,procesador.SLL_Range);
%     w_d = taylorwin(senal.npulsos,4,procesador.SLL_Doppler);
%     
%     ranged_Mrd = fft(w_r.'.*shaped_Mrd, senal.PRI*system.fs, 2);
%     
%     doppler_Mrd = fft(w_d.*ranged_Mrd, senal.npulsos, 1);
    
    %% Enventanado a cada fila y columna por separado 
    
    w_r = taylorwin(senal.PRI*system.fs,7,procesador.SLL_Range);
    w_d = taylorwin(senal.npulsos,7,procesador.SLL_Doppler);
    
    ranged_Mrd = fftshift(fft(bsxfun(@times, w_r.', demod_Mrd), senal.PRI*system.fs, 2) , 2);
    
    
    
    doppler_Mrd = fftshift(fft(bsxfun(@times, w_d, ranged_Mrd), senal.npulsos, 1), 1);

%% Con eventanado
    %Mrd = 20*log10(abs(doppler_Mrd)); % En escala logarítmica
     Mrd = abs(doppler_Mrd);

%% Enventanado estrecho
    
%      w_r = taylorwin(senal.npulsos,4,procesador.SLL_Range);
%      w_d = taylorwin(senal.npulsos,4,procesador.SLL_Doppler);
%      
%      ranged_Mrd = fft(w_r.*shaped_Mrd, senal.npulsos, 2);
%      
%      doppler_Mrd = fft(w_d.*ranged_Mrd, senal.npulsos, 1);    
%    
%      Mrd = abs(doppler_Mrd);
    
    %% Sin enventanado
%     ranged_Mrd = fft(shaped_Mrd,senal.PRI*system.fs, 2);
%     doppler_Mrd = fft(ranged_Mrd, senal.npulsos, 1);
%     Mrd = 20*log(abs(doppler_Mrd));

%% Cálculo rango y velocidad


nMues= senal.PRI*system.fs; 
eje_R = -senal.c*system.fs*(-nMues/2: nMues/2 -1)/(nMues*senal.gamma*2); 

eje_vel = -senal.lambda*(-senal.npulsos/2: senal.npulsos/2 -1)/(senal.PRI*senal.npulsos*2); 

%     figure
%    imagesc(eje_R, eje_vel, 20*log10(Mrd)); % 
%    colormap(jet);
%    colorbar;

  