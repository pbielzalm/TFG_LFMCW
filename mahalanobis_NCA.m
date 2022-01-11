function [detections_to_track, trazas, trazas_muertas, historial, promoEstablecida, conjuntoCovarianzas] = mahalanobis_NCA(interpoledPlot, trazasFiltradas, conjuntoCovarianzas, trazas, historial, nburst, config, P_ini, delta_t, system, senal)

%Inicializamos los parámetros
vuelta_umbral = 100;
umbralDrop = 3; % Calidad mínima
numExtrapConf = 3; % Extrapolaciones máximas
promoEstablecida = 5; %Calidad para pasar a ser firme
calidadMax = 10; % Calidad máxima
%promoTent = 2; % Veces seguidas ploteado para pasar a tentativa

plot_len = length(interpoledPlot.r);
interpoledPlot.no_corr=ones(1,plot_len);



detections_to_track = [];
trazas_muertas = [];

[unambig_velocity] = ambig2unambig (trazas, senal);

dthMah=15.6^2; %Dos dimensiones: (nz=2) --> Pg=0.995-> dth=10.6
noMah = 300;%Distancia para las vuelta_umbral primeras vueltas 

longTrazas=length(trazasFiltradas); %Longitud de  trazas

dist_aux = -1*ones(1,longTrazas);
plot_index_corr = ones(1,longTrazas);

trazas.no_corr = ones(1,longTrazas);

% Calculamos la aceleración vista
[est_acceleration] = acceleration_est (interpoledPlot, historial, delta_t, nburst, system);


%Comprobamos plot a plot si correla con alguna traza 
for i=1:plot_len
    distMin=inf; %Distancia mínima para ver a qué plot se acerca más
    trazaCorr=0; %posición del plot al que más se acerca
    dth = dthMah;
    dth_aux = noMah;
%         disp('vuelta de paneo');
    for k=1:longTrazas
        
        if ~((trazas.estado(k)>2) && (nnz(historial.trazas.r(:,k))>vuelta_umbral))% Si la traza no es firme y tiene menos de vuelta_umbral vueltas de vida

        Y = interpoledPlot.r(i);
        X =cell2mat(trazasFiltradas(k));
        X = X(1);
        distMah = abs(X - Y); % no es estrictamente distancia de mahala
      
        if(distMah<distMin)&& ~(distMin<dth) %Para comprobar mínima 
            distMin=distMah; 
            trazaCorr=k;
            dth = -inf;
            dth_aux = noMah;%Distancia para las vuelta_umbral primeras vueltas 
        end
        % A partir de vuelta_umbral, usar también la información en
        % distancia
        else

            Y = [interpoledPlot.r(i); interpoledPlot.v(i); est_acceleration(i)];
            X =cell2mat(trazasFiltradas(k));
            X(2) = unambig_velocity(k);
            difObs = X - Y;
            P = cell2mat(conjuntoCovarianzas(k));
            distMah=transpose(difObs)*inv(P)*difObs; %Fórmula Mahalanobis
%        distMah=transpose(difObs)*(P)\difObs; %Fórmula Mahalanobis 
        if distMah < 0
            disp('debug');
        end

            if(distMah<distMin) %Para comprobar mínima 
                distMin=distMah; 
                trazaCorr=k;
                dth_aux = -inf;
                dth = dthMah;
            end
        end
    end
    if (distMin<dth) || (distMin<dth_aux)  %Si baja el umbral (correla), se coloca en la traza con la que correla
        %% Si vuelve a correlar, comprobar si correla mejor.
        % Si es la primera vez que correla
        if (dist_aux(trazaCorr)<0)
            dist_aux(trazaCorr) = distMin; % Actual distancia del plot con el que mejor correla
            
            detections_to_track.r(trazaCorr)=interpoledPlot.r(i);
            detections_to_track.v(trazaCorr)=interpoledPlot.v(i);
            detections_to_track.a(trazaCorr)=0;
            
            interpoledPlot.no_corr(i)=0; % 0 si SÍ correla
            plot_index_corr(trazaCorr) = i; %Guardamos en la columna de la traza el índice del plot que ha correlado 
            trazas.no_corr(trazaCorr)=0; % 0 si SÍ correla
            % Aumentamos calidad si no ha llegado al límite
            if (trazas.calidad(trazaCorr)<(calidadMax))
                trazas.calidad(trazaCorr)= trazas.calidad(trazaCorr)+ 1;
            end
            trazas.extrapol_counter(trazaCorr) = 0;
            if (trazas.estado(trazaCorr)==1) && (trazas.calidad(trazaCorr)>umbralDrop) % Si era nueva, pasar a tentativa
                trazas.estado(trazaCorr)=2;
            elseif (trazas.estado(trazaCorr)==2) && (trazas.calidad(trazaCorr)>(promoEstablecida-1)) % Si era tentativa, pasar a firme
                trazas.estado(trazaCorr)=3;
            end
            % Se actualiza el historial
            historial.trazas.r(nburst, trazaCorr) = trazas.r(trazaCorr);
            historial.trazas.v(nburst, trazaCorr) = trazas.v(trazaCorr);
            historial.trazas.a(nburst, trazaCorr) = 0;
            

            historial.raw.r(nburst, trazaCorr) = interpoledPlot.r(i);
            historial.raw.v(nburst, trazaCorr) = interpoledPlot.v(i);
            historial.raw.a(nburst, trazaCorr) = 0;
            
        % Si no correla mejor, no hacer nada
        elseif (dist_aux(trazaCorr) < distMin)           
            continue
        % Si correla mejor y no es la primera que correla, cambiar el
        % plot asociado y la nueva distancia mínima
        else  
            dist_aux(trazaCorr) = distMin; % Actual distancia del plot con el que mejor correla
            detections_to_track.r(trazaCorr)=interpoledPlot.r(i);
            detections_to_track.v(trazaCorr)=interpoledPlot.v(i);
            detections_to_track.a(trazaCorr)=est_acceleration(i);
            interpoledPlot.no_corr(i)=0;
            % Se actualiza el historial
            interpoledPlot.no_corr(plot_index_corr(trazaCorr))=1; %Volvemos a establecer el plot como no correlado
            historial.trazas.r(nburst, trazaCorr) = trazas.r(trazaCorr);
            historial.trazas.v(nburst, trazaCorr) = trazas.v(trazaCorr);
            historial.trazas.a(nburst, trazaCorr) = trazas.a(trazaCorr);

            historial.raw.r(nburst, trazaCorr) = interpoledPlot.r(i);
            historial.raw.v(nburst, trazaCorr) = interpoledPlot.v(i);
            historial.raw.a(nburst, trazaCorr) = est_acceleration(i);
        
        end
    end
end
%% Si alguna traza no correla con ningún plot, bajamos la calidad (y extrapolar después)
if nnz(trazas.no_corr)>0
    for uncor = find(trazas.no_corr>0)
        trazas.calidad(uncor)= trazas.calidad(uncor) - 1; 
        trazas.extrapol_counter(uncor) = trazas.extrapol_counter(uncor) + 1;
        % Si no cumplen los requisitos, apuntar los índices para eliminar % más tarde
        if(uncor<1)
            disp("debug wewe");
        end
        
        if (((trazas.calidad(uncor)<umbralDrop) && trazas.estado(uncor) ~= 2) || (trazas.extrapol_counter(uncor) > numExtrapConf) || ((trazas.calidad(uncor)<1) && trazas.estado(uncor) == 2))
            trazas_muertas = cat(2, trazas_muertas, uncor);
            detections_to_track.r(uncor)=inf;
            detections_to_track.v(uncor)=inf;
            detections_to_track.a(uncor)=inf;
        else
            % Si cumple los requisitos, establecemos la NO detección
            detections_to_track.r(uncor)=0;
            detections_to_track.v(uncor)=0;
            detections_to_track.a(uncor)=0;
            
            historial.trazas.r(nburst, uncor) = trazas.r(uncor);
            historial.trazas.v(nburst, uncor) = trazas.v(uncor);
            historial.trazas.a(nburst, uncor) = trazas.a(uncor);
            
            % Se extrapola con la información anterior (historial.raw)
%             historial.raw.r(nburst, uncor) = historial.raw.r(nburst-1, uncor)+historial.raw.v(nburst-1, uncor)*delta;
%             historial.raw.v(nburst, uncor) = historial.raw.v(nburst-1, uncor);
            historial.raw.r(nburst, uncor) = 0;
            historial.raw.v(nburst, uncor) = 0;
            historial.raw.a(nburst, uncor) = 0;
        end
    end
end

%% Las detecciones nuevas se guardan y pasan a ser trzas potenciales
trazas.r = cat(2, trazas.r, interpoledPlot.r(interpoledPlot.no_corr>0)); % Añadir los plots NO correlados
trazas.v = cat(2, trazas.v, interpoledPlot.v(interpoledPlot.no_corr>0));
trazas.a = cat(2, trazas.a, est_acceleration(interpoledPlot.no_corr>0));

detections_to_track.r = cat(2, detections_to_track.r, interpoledPlot.r(interpoledPlot.no_corr>0));
detections_to_track.v = cat(2, detections_to_track.v, interpoledPlot.v(interpoledPlot.no_corr>0));
detections_to_track.a = cat(2, detections_to_track.a, est_acceleration(interpoledPlot.no_corr>0));

trazas.estado = cat(2, trazas.estado, ones(1, nnz(interpoledPlot.no_corr))); %Pasa a ser traza potencial
trazas.calidad=cat(2, trazas.calidad, ones(1, nnz(interpoledPlot.no_corr))*umbralDrop);
trazas.extrapol_counter=cat(2, trazas.extrapol_counter, zeros(1, nnz(interpoledPlot.no_corr)));


conjuntoCovarianzas_aux = cell(1,nnz(interpoledPlot.no_corr));
[conjuntoCovarianzas_aux{:}] = deal(P_ini);
conjuntoCovarianzas = cat(2,conjuntoCovarianzas, conjuntoCovarianzas_aux);
%Historial
columnas_zeros_historial_aux = zeros(config.num_bursts, nnz(interpoledPlot.no_corr));

columnas_zeros_historial_aux(nburst, :) = interpoledPlot.r(interpoledPlot.no_corr>0);
historial.trazas.r = cat(2, historial.trazas.r, columnas_zeros_historial_aux);
historial.raw.r = cat(2, historial.raw.r, columnas_zeros_historial_aux);

historial.raw.a = cat(2, historial.raw.a, columnas_zeros_historial_aux);
historial.trazas.a = cat(2, historial.trazas.a, columnas_zeros_historial_aux);
        
columnas_zeros_historial_aux(nburst, :) = interpoledPlot.v(interpoledPlot.no_corr>0);
historial.trazas.v = cat(2, historial.trazas.v, columnas_zeros_historial_aux);
historial.raw.v = cat(2, historial.raw.v, columnas_zeros_historial_aux);


% Borramos las detecciones de las trazas muertas
detections_to_track.r(detections_to_track.r == inf)=[];
detections_to_track.v(detections_to_track.v == inf)=[];
detections_to_track.a(detections_to_track.a == inf)=[];
